# GLM-5.3 EXL3 TP4：形状限定の推論最適化

対象はfull GLM-5.3の同一EXL3 checkpoint、TP4、FP8 KV。量子化方式、選択key数、K split、warp和、FWHT、cast、rank順FP32和は変更していない。Flashモデルへの置換ではない。

## 変更

- 非空のexpert memberタイルをGPU上で整数prefixにより一覧化し、既存GEMMへlogical座標を渡す。grouping・member順・small/big分類は保持する。M1/2は従来経路。
- sparse Decode：HBT4、4 warps、3 stages。R1/2、16 heads、512 latent、64 RoPE、FP8 KV、2048 selections、検証済みbucketだけを対象にする。
- sparse Prefill：HBT16、4 warps、3 stages。32/128/512/2048行の検証済み形状だけを対象にする。
- dense Decode：HBT4、4 warps、3 stages。R1/2のfull GLM FP8形状だけを対象にする。
- 厳密selectorの32行Prefill slab。保存FP32 scoreに対する閾値、小ID優先、最終ID昇順、private容量clampを保持する。
- EXL3 G2：独立した2つの16行タイルでtrellis復号を再利用する。採用dispatchは32/64/128行、4/5/6bit mul1、既存mode7。各rowのMMA・K順、counterのpass位置は変更しない。G1の算術は元のまま。

すべてopt-in。未知の形状は従来へ戻し、rankごとのautotuneは行わない。policy・layout・source/ABIをrank namespaceへ含める。

```bash
TF_GLM_GROUPED_MODE=transpose
TF_GLM_LINEAR_MODE=transpose_narrow
TF_GLM_ROUTE_MODE=parallel
TF_GLM_SELECTOR_MODE=parallel
TF_GLM_COMPACT_TILES=1
TF_GLM_MLA_POLICY=decode443
TF_GLM_PREFILL_MLA_POLICY=prefill1643
TF_GLM_DENSE_POLICY=decode443
TF_GLM_PREFILL_SELECTOR=parallel
TF_EXL3_PREFILL_G2=1
TF_EXL3_PAIR_COMPACT=0
TF_GLM_PROFILE=0
```

`PROFILE=0`は同期型計測の運用負荷を外す設定であり、kernel改良の利得とは別。

## 実モデルの同条件A/B

SM120の4 GPU、C1、MTP1、context=229376。同じ保存入力、greedy、最大256生成token。各条件ABBAを2 block、各方式4応答。EOSで終了した実token数を使用し、Decodeは最初のPrefill tokenを除いて計算した。Prefillは全て`cached=0`。

Aは同一測定binaryの従来dispatch、Bは最終組合せ。表は最初の新形状の初期費用も含む平均。

| 入力 | cold Prefill A→B | Decode A→B |
|---|---:|---:|
| 日本語4,096 | 6.083→4.592 s | 83.86→92.73 tok/s |
| code 16,384、保留入力 | 27.253→19.106 s | 82.98→91.92 tok/s |
| 自然文65,536 | 120.039→82.152 s | 76.59→84.64 tok/s |
| code 76,000 | 141.595→93.933 s | 74.69→82.72 tok/s |
| 自然文131,072 | 264.800→172.861 s | 72.67→80.57 tok/s |
| 自然文228,800 | 523.738→324.562 s | 74.09→82.07 tok/s |

上限近くの候補の最初のPrefillは355.078 s、後続3回は314.370/314.360/314.440 s。初回費用を隠して定常速度としない。初期費用には未キャッシュのkernel準備が含まれ得る。

凍結した**元ソース・元native binary**との別cold対照でも、元ID列が一致し、76KのPrefillは140.242→93.359 s、Decodeは約75.44→83.41 tok/sだった。こちらは各2応答、プロセス別A/Bであり、ABBA系列と区別する。

## 検証

- C1の263応答、固定seed通常sampling、serial、cold/prefix hit、main/MTPの実partial・出力68比較。
- C2/4のsolo/packed・持続C4・長短混在224応答。windowは同じ32768で比較し、229KのC1とは混ぜない。
- 元ソース対照12応答。元ID列は全系列で一致。
- GLM CPU回帰342 passed / 12 skipped。
- 最新統合native資格419 passed。別revisionの件数とは合算しない。
- 独立整数oracle、4数値分布、複数shape、canary、changed-input graph replay。attentionの既存FP64 oracleも再実行。
- 限定memcheck/racecheckはエラー0。全shapeのsanitizer保証ではない。

C4持続負荷は各100要求、測定期間260.82/244.33 s、aggregate 98.15→104.78 tok/s。request latency P95は11.140→10.451 s。prefix再利用がある系列で、cold Prefill改善とは呼ばない。P99は各100標本しかなく精度が低い。C4はプロセス別A/Bで、CIのあるABBAとは区別する。

## 不採用・制限

- pairの空CTAを336→208へ詰める案は、実重みで横ばい〜退行。配信では無効。
- PF増加、一律NT4、N32、L2 hint、旧bank swizzleは再採用していない。
- 最初のJIT/capture、workspace、prefix条件を分離する。64-token境界試験を長い生成の代表値としない。
- 6種類には自然文・code・日本語と、合成の数学・会話履歴・JSON/logが含まれる。実利用分布の代表性や、全入力での高速化を保証しない。
- 元ID不変は回答品質の保証ではない。長文参照問題やコード実行による独立の品質評価は未完了。
- NCUのDRAM/stall/achieved occupancy、native NCCLの純転送と到着待ち、最終4rank traceは未完了。静的register数をoccupancyと呼ばない。
- 新しいprefix共有allocator、Schedulerの優先順位変更、dequant staging、persistent schedulingは採用していない。

## 一次資料

- [CUTLASS grouped scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)：device側tile割当の参考。EXL3をCUTLASSへ置換していない。
- [Triton range](https://triton-lang.org/main/python-api/generated/triton.language.range.html)、[issue #6691](https://github.com/triton-lang/triton/issues/6691)：pipelineを正しさ検査なしに変更しない。
- [CUDA register spilling](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)：register・stack・sharedと実時間を区別する。
- [KernelBench](https://arxiv.org/abs/2502.10517)：固定shapeの単発allcloseだけに依存しない。
