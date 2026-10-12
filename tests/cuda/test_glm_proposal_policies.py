"""Target IDs and private State remain exact across calibrated/adaptive native proposals."""
import os
import pytest
import torch
from test_glm_segmented import weights
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

@pytest.mark.parametrize("policy",["t7:0.1:0.7","t7:0.1:1.3","g7:0.7:0.16"])
@pytest.mark.parametrize("greedy",[False,True])
def test_calibrated_and_gain_policy_target_ids_prefix_cancel(weights, policy, greedy):
    from tensorfold.engine.exact_sampling import Sampling
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder
    class Comm:
        world = 1
        def all_gather(self, send, recv):
            recv.copy_(send.reshape(-1))
    w = weights
    w.comm = Comm()
    engine = object.__new__(GlmEngine)
    engine.w, engine.rank, engine.world, engine.comm, engine.torch = w, 0, 1, w.comm, torch
    engine.full, engine.limit, engine.parallel, engine.concurrent = True, 2112, 4, True
    engine.policy, engine.serial_only, engine.drafter = policy, False, None
    engine.eos, engine.request, engine.cache_entries = w.cfg.eos, threading.local(), 4
    engine.capacity_plan = {"budget_bytes": int(torch.cuda.mem_get_info()[1] * .85)}
    engine.e = Engine(w, capacity=2120, max_rows=8, prefill_rows=128, graphs=True, long_context=True)
    engine.multi = MultiDecoder(engine)
    engine.scheduler = Scheduler(engine.multi, max_streams=4)
    prompts = [[10 + i] * (33 + i) for i in range(5)]
    samplings=[None if greedy else Sampling(9530+i,1.,0,.95) for i in range(5)]
    expected = {}
    try:
        for i, prompt in enumerate(prompts):
            out = []
            engine.generate(prompt, 16, samplings[i], lambda t: out.extend(t), draft=False)
            expected[i] = out
        barrier = threading.Barrier(5)
        peak = [0]
        original = engine.multi._admit_local
        def tracked(*a, **kw):
            result = original(*a, **kw)
            peak[0] = max(peak[0], engine.multi.live())
            return result
        engine.multi._admit_local = tracked
        def run(i):
            out = []
            barrier.wait()
            stats = engine.generate(prompts[i], 16, samplings[i], lambda t: out.extend(t))
            return out, stats
        with ThreadPoolExecutor(5) as pool:
            jobs = [pool.submit(run, i) for i in range(5)]
            results = [f.result(timeout=float(os.environ.get("TF_TEST_TIMEOUT_S", "90"))) for f in jobs]
        assert 2 <= peak[0] <= 4
        assert [r[0] for r in results] == [expected[i] for i in range(5)]
        assert not engine.multi.live()
        out = []
        stats = engine.generate(prompts[0], 16, samplings[0], lambda t: out.extend(t))
        assert out == expected[0] and stats["cached"] > 0
        stopped = []
        engine.generate(prompts[0], 64, samplings[0], lambda t: (stopped.extend(t), True)[1])
        assert len(stopped) < 64
        out = []
        engine.generate(prompts[0], 16, samplings[0], lambda t: out.extend(t))
        assert out == expected[0]
    finally:
        engine.close()
        w.comm = None
