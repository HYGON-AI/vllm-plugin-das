# Split Adapter Distribution Plan

> Use superpowers:executing-plans inline.

Goal: fix the Plugin host dependency and installed ownership checks for current Adapter dual wheels without changing numerical runtime.

1. RED dependency and built-wheel metadata contracts.
2. RED real-stdlib installed-origin fixtures for modern split and mixed ownership; retain legacy acceptance.
3. Implement metadata dependency and shared stdlib owner selector, integrate verifier/CI guard and update README.
4. Run affected unit/source tiers; build current upstream split wheels and isolated install/discovery proof with correct origins and entrypoint ownership.
5. Independent final source/evidence review, fix material findings, fresh remote duplicate check, then submit applicable fork MR.

Do not change existing model runtimes, source pins or CI image/authorization protections.
