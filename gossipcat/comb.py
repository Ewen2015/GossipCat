from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Feature(object):

	def __init__(self, data, target, features):
		"""
		Args:
			data: A dataset you wanna play with.
			target: The target name of your dataset.
			features: The feature names of your dataset.
		"""
		self.data = data
		self.target = target
		self.features = features

		self.predictors = []
		self.dup = []
		self.int_lst, self.float_lst, self.object_lst = [], [], []
		self.corr_lst = []
		self.new_data_comb = pd.DataFrame()
		self.new_col_comb = []
		self.result_COM = pd.DataFrame()
		self.result_LDA = pd.DataFrame()


	def generate_comb(self, features_max = 100, kill = 50, n_combinations = 2, auc_score = 0.7):

		X = self.data[self.features]
		y = self.data[self.target]

		features_temp = X.columns

		self.result_COM = pd.DataFrame(columns=['Features', 'AUC', 'Coefficients', 'Intercept'])
		self.new_data_comb = pd.DataFrame()

		while (len(features_temp) > features_max):
		    
		    lda = LinearDiscriminantAnalysis(n_components=2)
		    temp = StandardScaler().fit_transform(X[features_temp].fillna(X.median()))
		    lda.fit(temp, y)
		    self.result_LDA = pd.DataFrame({'title': X[features_temp].columns,
		                               'coff': np.abs(lda.coef_.reshape(X[features_temp].shape[1]))})
		    features_temp = list(self.result_LDA.sort_values(ascending=False, by=['coff']).iloc[:, 1].values)
		    del features_temp[-kill:]
		    print('Turns Left:', round(len(features_temp)/kill))

		print('\nNumber of features left: ', len(features_temp))

		features_comb = list(itertools.combinations(features_temp, n_combinations))

		for index, value in enumerate(features_comb):

		    golden_features = X[list(value)]
		    lda = LinearDiscriminantAnalysis(n_components=2)
		    temp = lda.fit_transform(golden_features.fillna(golden_features.median()), y)
		    prob = lda.predict_proba(golden_features.fillna(golden_features.median()))[:, 1]
		    auc = metrics.roc_auc_score(y, prob)

		    if auc > auc_score:
		        print('-'.join(value), ' AUC (train): ', auc)
		        self.new_data_comb['-'.join(value)] = temp.tolist()
		        self.result_COM=self.result_COM.append({'Features': value, 'AUC': auc,
		                                      'Coefficients': lda.coef_,'Intercept': lda.intercept_}, 
		                                     ignore_index=True)

		    if index % 200 == 0:
		        print('Turns Remaining: ', len(features_comb) - index)

		self.result_COM = self.result_COM.sort_values(by='AUC', ascending=False)

		return self.new_data_comb, self.new_data_comb.columns.tolist()
