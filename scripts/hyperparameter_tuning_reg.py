from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV


def train_model(x_train, y_train, eval_set, classif, params, cv_params):
   model_cv = RandomizedSearchCV(
       estimator=classif,
       param_distributions=params,
       **cv_params
   )


   print("--- Training model ---")
   model_cv.fit(x_train, y_train, eval_set=eval_set)


   print("--- Best model parameters ---")
   print(model_cv.best_params_)


   return model_cv.best_estimator_


hps_params = {
   "objective": ["binary"],
   "boosting_type": ["gbdt"],
   "n_estimators": [100],
   "lambda_l1": uniform(1e-8, 100.0),
   "lambda_l2": uniform(1e-8, 100.0),
   "max_depth": randint(2, 128),
   "num_leaves": randint(2, 512),
   "min_data_in_leaf": randint(10, 100000),
   "min_gain_to_split": loguniform(0.001, 1.0),
   "min_sum_hessian_in_leaf": loguniform(1e-5, 0.1),
   "max_bin": randint(2, 2048),
   "max_cat_threshold": randint(1, 100),
   "verbose": [-1],
   "verbose_eval": [-1]
}


cv_params = {
   'scoring': 'f1',
   'n_iter': 10,
   'n_jobs': 20
}
