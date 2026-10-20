# Security policy

## Supported versions

Security fixes are released for the latest minor version only.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

The container image `ghcr.io/imranmali/structured-output-agent` follows the
same policy: use the latest `0.2.x` tag (or `latest`), which is rebuilt on
every push to `main`.

## Reporting a vulnerability

Please do **not** open a public issue, pull request or discussion for a
security problem.

Report it privately through GitHub's private vulnerability reporting:

1. Go to the repository's **Security** tab and choose
   [**Report a vulnerability**](https://github.com/imranmali/structured-output-agent/security/advisories/new).
2. Describe the issue, the affected version or commit, and how to reproduce
   it (a minimal script, request or configuration). Say whether you believe it
   is already being exploited.

The report is visible only to you and the maintainers. We coordinate the fix
in a draft security advisory, credit you there unless you prefer otherwise,
and request a CVE when one is warranted.

## Response targets

These are targets for a small, volunteer-maintained project, not guarantees:

| Step                                   | Target                       |
| -------------------------------------- | ---------------------------- |
| Acknowledge the report                 | 3 business days              |
| Initial assessment and severity        | 7 days                       |
| Fix released for critical / high       | 30 days                      |
| Fix released for medium / low          | Next scheduled release       |
| Public advisory                        | When the fix is released     |

If a target slips we will tell you why and when to expect the next update.

## Scope and deployment notes

In scope: the `structured_output_agent` library, the `structured-agent` CLI
and HTTP service, the container image, the manifests in `deploy/`, and the
CI/CD workflows in this repository. Vulnerabilities in third-party
dependencies should be reported upstream; tell us too if this project's use
of them makes the issue exploitable.

Some risks come from how the library is used rather than from bugs in it.
Keep them in mind when you deploy:

- **Prompt injection via tool outputs.** Tool results and retrieved content
  are sent back to the model and can contain instructions aimed at it. Schema
  validation constrains the *shape* of tool arguments and final output, not
  their intent. Treat validated output as untrusted input to the rest of your
  system, keep tools least-privileged, and validate side-effecting tool
  arguments (amounts, paths, recipients) in the tool itself. Reports that
  validation can be bypassed, so that output that breaks the declared schema
  is returned as valid, are in scope.
- **Logging of model output.** Validation failure events record raw model
  output, which can contain personal data or secrets echoed from prompts or
  tools. Pass a redaction function, e.g. `ValidationLogger(redact=...)`, to
  scrub it before it is stored or emitted, and keep retention bounded with
  `max_events`.
- **Service authentication.** The HTTP service accepts API keys from the
  `SOA_API_KEYS` environment variable. Set it for any deployment reachable by
  untrusted clients, supply the keys from a secret store (for example a
  Kubernetes Secret) rather than baking them into images or manifests, and
  rotate them by deploying a new list. Health, readiness and metrics endpoints
  stay unauthenticated for orchestrators, so keep `/metrics` off the public
  internet (expose the service through an ingress that routes `/v1/*` only).
- **Provider credentials.** `ANTHROPIC_API_KEY` must come from the
  environment or a secret store. Never log it, never include it in error
  reports or issues, and never pass it on the command line where it ends up in
  shell history or process listings. If you believe it has been exposed,
  revoke it in the Anthropic console first, then report the leak path to us
  if it was caused by this project.
