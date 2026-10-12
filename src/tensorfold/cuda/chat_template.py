"""The CUDA server's chat template: the checkpoint's own Jinja template, rendered as Hugging Face does."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorfold.server.messages import _normalize_tool_call_arguments, late_system_role, normalize_messages
from tensorfold.server.request_options import effort_levels


class ChatTemplate:
    """The model's own Jinja chat template, rendered the way Hugging Face's apply_chat_template does."""

    def __init__(self, model_dir: Path):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        source_path = model_dir / "chat_template.jinja"
        source = source_path.read_text() if source_path.exists() else cfg["chat_template"]

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(source)
        # GLM-5.3's template never mentions ``enable_thinking``: it always opens the thought block itself, so a
        # request that asks for no thinking cannot be honoured by the prompt (the server would hand the thought back
        # as the answer). The server reads this and treats such a reply as thinking.
        self.thinking_switch = "enable_thinking" in source
        self.efforts = effort_levels(source)          # the reasoning efforts it names (the Mac reads the same ones)
        self.specials = {k: (v.get("content") if isinstance(v, dict) else v)
                         for k, v in cfg.items() if k in ("bos_token", "eos_token", "pad_token", "unk_token")}
        self.late_system = late_system_role(
            lambda messages: self.template.render(**self.specials, messages=messages, add_generation_prompt=False))

    @staticmethod
    def without_effort_system(text: str) -> str:
        """The rendered prompt without the ``<|system|>Reasoning Effort: …`` segment its template prepends.

        GLM-5.3's template writes that line for every request and cannot drop it through a parameter. Measured on this
        checkpoint, the line puts the model into a rambling thought instead of an answer (the same prompt without it
        answers directly), so a client that asked for no thinking gets a prompt without it.
        """

        at = text.find("<|system|>Reasoning Effort:")
        if at < 0 or at > 64:                              # only the leading segment the template always writes
            return text
        after = text.find("<|user|>", at)
        if after < 0:
            after = text.find("<|system|>", at + 1)
        return text[:at] + text[after:] if after > 0 else text

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
               enable_thinking: bool, extra: dict[str, Any] | None = None, allow_images: bool = False) -> str:
        messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=self.late_system,
                                                                     allow_images=allow_images))
        kwargs = dict(self.specials, messages=messages, tools=tools or None, add_generation_prompt=True,
                      enable_thinking=enable_thinking)
        kwargs.update(extra or {})
        return self.template.render(**kwargs)
