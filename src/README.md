# Source code

This folder contains our implementation of DIME and standard MLP classifiers for experiments.

## Code Attribution

The following files in this directory reuse and adapt code from the original DIME implementation:

- `cmi_estimator_prior.py`
- `cmi_estimator.py`
- `dime_utils.py`
- `masking_pretrainer_prior.py`
- `masking_pretrainer.py`
- `models.py`

Original source: [https://github.com/suinleelab/DIME](https://github.com/suinleelab/DIME)

### License

The original DIME implementation is licensed under the Apache License 2.0. Our adaptations and extensions are also provided under the same license.

### References

For more information on the original methodology, please see:
- Original paper: 
- Original repository: [https://github.com/suinleelab/DIME](https://github.com/suinleelab/DIME)

## Modifications

Our implementation extends the original work with the following modifications:
- rework of the code for newer Python, Pytorch and PytorchLightning versions,
- helper methods,
- formatting.

---

*This implementation complies with the Apache License 2.0 terms for code reuse and attribution.*