from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
from scipy.stats import poisson


class StaticCrimeMLEModel(GenericLikelihoodModel):
	"""Model to compute lambdas using MLE for the static crime simulation"""

	def __init__(self, endog, exog=None, **kwds):
		"""Initializes the model

		Args:
			endog (np.ndarray): the observed crimes and allocated patrols
			exog (np.ndarray, optional): observed variables from outside our system, not used in our case. 
									   Defaults to None.
		"""
		if exog is None:
			exog = np.zeros_like(endog)

		super(StaticCrimeMLEModel, self).__init__(endog, exog, **kwds)

		self.data.xnames = ["x1"]

	def nloglikeobs(self, params):
		"""Computes the negative log likelihood of the model

		Args:
			params (np.ndarray): the parameters of the model 
							  (contains only initial lambda of the area in our case)

		Returns:
			float: the negative log likelihood
		"""
		lambda_ = params[0]

		ll_output = self._myloglikelihood(self.endog, lambda_=lambda_)

		return -np.log(ll_output)

	def fit(self, start_params=None, maxiter=10000, **kwds):
		"""Fits the model using MLE"""
		if start_params is None:
			lambda_start = self.endog[:, 0].mean()
			start_params = np.array([lambda_start])

		return super(StaticCrimeMLEModel, self).fit(start_params=start_params, maxiter=maxiter, **kwds)

	def _myloglikelihood(self, data, lambda_):
		"""Computes the log likelihood for each one of the observed datapoints given the parameter 
		   lambda of the distribution

		Args:
			data (np.ndarray): the data containing observed crimes and allocated patrols
			lambda_ (int): lambda parameter of the poisson distribution

		Returns:
			np.ndarray: array containing the log likelihood for each one of the observed datapoints
		"""
		obs_crimes = np.array(data[:, 0])
		allocated_patrols = np.array(data[:, 1])
		output_array = np.zeros(len(obs_crimes))

		assert len(obs_crimes) == len(allocated_patrols)

		output_array += (obs_crimes < allocated_patrols) * poisson.pmf(obs_crimes, lambda_)

		# The probability of observing a count equal to the allocated count is
		# the tail of the poisson pmf from the obs_crimes value.
		output_array += (obs_crimes == allocated_patrols) * (1.0 - poisson.cdf(obs_crimes - 1, lambda_))

		return output_array