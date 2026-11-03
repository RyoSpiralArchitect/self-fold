"""Native Gemma source features and a continuous interface to the ordinary A/B token."""

import hashlib

import torch
import torch.nn.functional as F

from selffold.prompts import INPUT_MARKER
from selffold.records import digest


def account(host, tokens):
    host.meter.forwards += 1
    host.meter.processed_tokens += tokens


@torch.no_grad()
def capture_clauses(host, texts):
    """One frozen pass per batch of raw clauses. No prompt, later world or option labels."""
    result = {}
    for start in range(0, len(texts), 16):
        batch = texts[start : start + 16]
        seq = [host.tokenizer.encode(s, add_special_tokens=True) for s in batch]
        width = max(map(len, seq))
        if width > host.max_context or any(not r for r in seq):
            raise ValueError("Invalid source capture length")
        ids = torch.tensor(
            [r + [host.tokenizer.pad_token_id] * (width - len(r)) for r in seq],
            device=host.device,
        )
        lengths = torch.tensor([len(r) for r in seq], device=host.device)
        mask = torch.arange(width, device=host.device)[None, :] < lengths[:, None]
        host._budget(ids.numel())
        output = host.model.model(
            input_ids=ids,
            attention_mask=mask.long(),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = output.hidden_states[host.cut].float()
        last = hidden[torch.arange(len(batch), device=host.device), lengths - 1]
        mean = (hidden * mask[..., None]).sum(1) / lengths[:, None]
        features = torch.cat(
            [F.layer_norm(x, (host.width,)) for x in (last, mean)], -1
        ).cpu()
        result.update(zip(batch, features, strict=True))
        account(host, ids.numel())
    return result


class NativeChoice:
    """Only a continuous action row and two displayed options enter the final LM call."""

    def __init__(self, host, menus):
        self.host = host
        self.labels = [self.single(k) for k in ("A", "B")]
        self.prompts = {}
        for group, menu in enumerate(menus):
            for order in (0, 1):
                text = (
                    f"My stored action is {INPUT_MARKER}.\n"
                    f"Which option names my stored action?\nA = {menu[order]}. B = {menu[1-order]}.\n"
                    "Respond with only A or B."
                )
                ids, where = host.render(text)
                self.prompts[group, order] = (ids, where[INPUT_MARKER])
        phrases = [
            host.tokenizer.encode(s, add_special_tokens=False)
            for menu in menus
            for s in menu
        ]
        if max(map(len, phrases)) > host.slots or min(map(len, phrases)) < 1:
            raise ValueError("An action exceeds the fixed lexical bank width")
        blank = self.single(" ")
        tokens = [[blank] * (host.slots - len(p)) + p for p in phrases]
        with torch.no_grad():
            self.bank = (
                host.model.get_input_embeddings()(
                    torch.tensor(tokens, device=host.device)
                )
                .float()
                .detach()
            )
        self.binding = dict(
            tokens=tokens,
            labels=self.labels,
            prompts_sha256={
                f"{m}/{o}": digest(ids) for (m, o), (ids, _) in self.prompts.items()
            },
        )

    def single(self, text):
        tokens = self.host.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Need one native token for {text!r}")
        return tokens[0]

    def lexical_parity(self, reader):
        """Compare the mixed one-hot bank with the actual phrase token IDs at the same positions."""
        device = self.host.device
        phrases = torch.tensor(self.binding["tokens"], device=device)
        action = 2
        ids, pos = self.prompts[1, 1]
        literal = torch.tensor([ids], device=device)
        literal[0, pos] = phrases[action]
        self.host._budget(literal.numel())
        with torch.no_grad(), reader.active():
            reference = (
                self.host.model(
                    input_ids=literal,
                    attention_mask=torch.ones_like(literal),
                    position_ids=torch.arange(len(ids), device=device)[None],
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits[:, -1]
                .float()
            )
        account(self.host, literal.numel())
        row = F.one_hot(torch.tensor([action]), len(phrases)).float()
        with torch.no_grad():
            return torch.equal(reference, self.logits(reader, row, [1], [1]))

    def logits(self, reader, weights, menus, orders):
        host = self.host
        contexts = [
            self.prompts[int(m), int(o)] for m, o in zip(menus, orders, strict=True)
        ]
        width = max(len(ids) for ids, _ in contexts)
        ids = torch.tensor(
            [[host.tokenizer.pad_token_id] * (width - len(r)) + r for r, _ in contexts],
            device=host.device,
        )
        pos = torch.tensor(
            [[p + width - len(r) for p in pp] for r, pp in contexts], device=host.device
        )
        lengths = torch.tensor([len(r) for r, _ in contexts], device=host.device)
        mask = (
            torch.arange(width, device=host.device)[None, :] >= width - lengths[:, None]
        )
        soft = (weights.to(host.device) @ self.bank.flatten(1)).reshape(
            -1, host.slots, host.width
        )

        def replace(module, args, output):
            h = output.clone()
            h[torch.arange(len(h), device=host.device)[:, None], pos] = soft.to(h.dtype)
            return h

        host._budget(ids.numel())
        hook = host.model.get_input_embeddings().register_forward_hook(replace)
        try:
            with reader.active():
                answer = host.model(
                    input_ids=ids,
                    attention_mask=mask.long(),
                    position_ids=(mask.long().cumsum(1) - 1).clamp_min(0),
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            if answer.past_key_values is not None:
                raise RuntimeError("Unexpected KV state")
            logits = answer.logits[:, -1].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Invalid native scores")
            account(host, ids.numel())
            return logits
        finally:
            hook.remove()


def fingerprint(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
