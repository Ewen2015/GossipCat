import matplotlib.pyplot as plt
import seaborn as sns

def BiBoxplot(target, features, data):
  for f in features:
    plt.figure(figsize=(16, 1))
    sns.boxplot(y=target, x=f, data=data, orient='h', width=0.4, fliersize=0.3)
    plt.show()
  return None

def BiDensity(target, features, data):
  targetList = data[target].unique().tolist()
  for f in features:
    plt.figure()
    for cat in targetList:
      ax = sns.kdeplot(data[f][data[target]==cat], shade=True, label=cat)
      ax.set(xlabel=f, ylabel='density')
      ax.legend(title=target)
  return None

