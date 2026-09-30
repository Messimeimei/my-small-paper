# CSI model identity and reasoning configuration

## Verified model versions and dates

The GitCode discussion `org/openCsiTool/discussions/1` was created on
2026-04-10 and last modified on 2026-09-22. Its current catalog lists these
selected aliases. Vendor documentation provides these identities:

| CSI request alias | Vendor model/version | Release or snapshot date | Evidence |
| --- | --- | --- | --- |
| `Qwen3.8-Max` | `qwen3.8-max`; current documented snapshot `qwen3.8-max-0902`, alias `qwen3.8-max-2026-09-02` | 2026-09-02 | Alibaba Model Studio model card |
| `DeepSeek-V4-Pro` | DeepSeek-V4-Pro GA; Alibaba catalog link `deepseek-v4-pro-0813` | 2026-08-13 | DeepSeek official GA release and Alibaba catalog |
| `GLM-5.3` | GLM-5.3; same base model as GLM-5.2 with additional post-training | 2026-08-14 | Z.ai official technical report |

Sources:

- <https://help.aliyun.com/zh/model-studio/qwen3-8-max>
- <https://api-docs.deepseek.com/news/news260813>
- <https://z.ai/blog/glm-5.3>
- <https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3>

CSI still exposes undated request aliases. The dates above identify the vendor
versions currently documented around those aliases; only CSI `/v1/models`, a
returned immutable model ID/fingerprint, or written CSI confirmation can prove
which snapshot the proxy actually served for a particular evaluation run.

## Reasoning strength

The CSI tutorial does not describe thinking parameters, but the vendor docs do.
The bundled `model_request_options.json` uses:

| Model | Thinking configuration | Vendor behavior |
| --- | --- | --- |
| `Qwen3.8-Max` | `enable_thinking=true`, `thinking_budget=8192` | Qwen Chat Completions uses `enable_thinking`; budget range is 1--32768 |
| `DeepSeek-V4-Pro` | `thinking.type=enabled`, `reasoning_effort=high` | Supports low/high/max; thinking defaults to enabled and high |
| `GLM-5.3` | `thinking.type=enabled`, `reasoning_effort=high` | Thinking cannot be disabled; supports low/high/max and defaults to max |

All requests use `max_tokens=8192`. For Qwen, `thinking_budget=8192` limits
reasoning separately while `max_tokens` limits the final response. DeepSeek
documents that `temperature` is ignored in thinking mode. The request retains
`temperature=0` to preserve the original audit request shape for other models.

The effective per-model fields are saved in every run manifest. They can be
disabled with `--disable-model-options` for a provider-default control.

## Probe on the company network

The metadata probe queries `/v1/models` and sends a 64-token request to each
selected model. With `--reasoning-matrix`, it tests the appropriate vendor
controls for each model, including the bundled high configuration:

```bash
export CSI_API_KEY='YOUR_KEY'
python probe_csi_models.py --reasoning-matrix
```

Inspect `csi_model_probe.json` for:

- the provider's exact `model` string;
- `system_fingerprint`, if exposed;
- catalog `created` and ownership metadata, if exposed;
- whether `reasoning_content` or reasoning-token usage is returned;
- HTTP acceptance or rejection of each model-specific thinking configuration.

A successful HTTP response only proves that a field was accepted; it may still
be ignored by a proxy. Exact backend dating requires an immutable revision or
written confirmation from the CSI service owner.
