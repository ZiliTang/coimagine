"""CoImagine 2D first round: private contact beliefs, finite-bit messages, exact oracles.

The package is organised as:

* :mod:`coimagine.protocol` -- strictly isolated information objects and the
  decision-boundary contract;
* :mod:`coimagine.toy` -- micro finite-state generative models and instances M0-M3;
* :mod:`coimagine.exact` -- Bayes filters, message codecs, exact oracles and a
  deliberately non-exact limited search;
* :mod:`coimagine.diagnostics` -- runs the micro oracle suite and writes the
  oracle-gap report.

The deprecated Gate A research line lives in ``_old/`` and must never be imported.
"""

__version__ = "0.1.0"
