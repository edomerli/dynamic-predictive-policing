from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
from scipy.stats import poisson


class StaticCrimeMLEModel(GenericLikelihoodModel):

	def __init__(self, endog, exog=None, **kwds):
		if exog is None:
			exog = np.zeros_like(endog)

		super(StaticCrimeMLEModel, self).__init__(endog, exog, **kwds)

    	# Setting xnames manually so model has correct number of parameters.
		self.data.xnames = ["x1"]

	def nloglikeobs(self, params):
		"""Return the negative loglikelihood of endog given the params for the model.

    	Args:
      		params: Vector containing parameters for the likelihood model.

    	Returns:
    		Negative loglikelihood of self.endog computed with given params.
		"""
		lambda_ = params[0]

		ll_output = self._myloglikelihood(self.endog, rate=lambda_)

		return -np.log(ll_output)

	def fit(self, start_params=None, maxiter=10000, **kwds):
		"""Override fit to call super's fit with desired start params."""
		if start_params is None:
			lambda_start = self.endog[:, 0].mean()
			start_params = np.array([lambda_start])

		return super(StaticCrimeMLEModel, self).fit(start_params=start_params, maxiter=maxiter, **kwds)

	def _myloglikelihood(self, data, rate):
		"""Return likelihood of the given data with the given rate as the poisson parameter."""
		obs_crimes = np.array(data[:, 0])
		allocated_patrols = np.array(data[:, 1])
		output_array = np.zeros(len(obs_crimes))

		assert len(obs_crimes) == len(allocated_patrols)

		output_array += (obs_crimes < allocated_patrols) * poisson.pmf(obs_crimes, rate)

		# The probability of observing a count equal to the allocated count is
		# the tail of the poisson pmf from the obs_crimes value.
		output_array += (obs_crimes == allocated_patrols) * (1.0 - poisson.cdf(obs_crimes - 1, rate))

		return output_array