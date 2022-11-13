
import dowhy
import dowhy.causal_estimators.linear_regression_estimator

def mediation_nde(model):
    identified_estimand_nde = model.identify_effect(estimand_type="nonparametric-nde", 
                                            proceed_when_unidentifiable=True)
    causal_estimate_nde = model.estimate_effect(identified_estimand_nde,
                                        method_name="mediation.two_stage_regression",
                                       confidence_intervals=False,
                                       test_significance=False,
                                        method_params = {
                                            'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
                                            'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
                                        }
                                       )
    return causal_estimate_nde.value

def mediation_nie(model):
    identified_estimand_nie = model.identify_effect(estimand_type="nonparametric-nie", 
                                                proceed_when_unidentifiable=True)
    causal_estimate_nie = model.estimate_effect(identified_estimand_nie,
                                            method_name="mediation.two_stage_regression",
                                           confidence_intervals=False,
                                           test_significance=False,
                                            method_params = {
                                                'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
                                                'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
                                            }
                                           )
    return causal_estimate_nie.value

def backdoor_ate(model):
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(identified_estimand,
            method_name="backdoor.linear_regression")
    return causal_estimate.value