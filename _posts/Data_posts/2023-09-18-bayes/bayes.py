import numpy as np
import matplotlib.pyplot as plt

class Bayes:

    def __init__(self):
        self.prior = {}
        self.mixes = {}
        self.data = {}

    def set(self, hypothesis, probability):
        self.prior[hypothesis] = probability
        return self.prior

    def normalise(self):
        total = sum(self.prior.values())
        list_of_hypos = list(self.prior.keys())
        for hypo in list_of_hypos:
            self.prior[hypo] = self.prior[hypo] / total
        return self.prior

    def mix(self, hypothesis, element_probability):
        self.mixes[hypothesis] = element_probability
        return self.mixes

    def update(self, data):
        self.data = data
        return self.data

    def likelihood(self):

        normalise_constant = []
        likelihoods = {}
        list_of_hypos = list(self.prior.keys())
        list_of_evidence = list(self.data.keys())
        for hypo in list_of_hypos:
            P_E_H = []
            for evidence in list_of_evidence:
                P_E_H.append(self.mixes[hypo][evidence]**self.data[evidence])
            P_E_H = np.prod(np.array(P_E_H))
            normalise_constant.append(P_E_H * self.prior[hypo])
            likelihoods[hypo] = P_E_H * self.prior[hypo]

        constant = np.sum(np.array(normalise_constant))
        return constant, likelihoods

    def posterior(self):
        constant, likelihoods = self.likelihood()
        list_of_hypos = list(self.prior.keys())
        posteriors = {}
        for hypo in list_of_hypos:
            posteriors[hypo] = likelihoods[hypo] / constant
        return posteriors

    def get_max(self):
        posteriors = self.posterior()
        return max(posteriors, key=posteriors.get)

    def credible_interval(self):
        posteriors = self.posterior()
        cdf = np.cumsum(np.array(list(posteriors.values())))

        percentile_5 = np.where(cdf >= 0.05)[0][0] + 1
        percentile_95 = np.where(cdf >= 0.95)[0][0] + 1

        print('The 5th percentile is: ' + str(percentile_5))
        print('The 95th percentile is: ' + str(percentile_95))

        return cdf


# %%

Cookie = Bayes()
Cookie.set('Bowl 1', 0.5)
Cookie.set('Bowl 2', 0.5)
mix1 = dict(van=0.75, choc=0.25)
mix2 = dict(van=0.5, choc=0.5)
Cookie.mix('Bowl 1', mix1)
Cookie.mix('Bowl 2', mix2)

data1 = dict(van=5, choc=0)
Cookie.update(data1)
Cookie.likelihood()
plt.hlines(0.5, -3, 3, colors='r', label='Prior')
plt.bar(Cookie.prior.keys(),Cookie.posterior().values(), label='Posterior')
plt.legend()
plt.show()

#%%

# create one hundred bowls, each with the same probability for Cookie.set

Cook = Bayes()
for i in range(1, 101):
    Cook.set('Bowl ' + str(i), 1/100)

# plot the prior
x = np.arange(1, 101)
y = Cook.prior.values()

for i in range(1,101):
    mix = dict(van=i/100, choc=(1-i/100))
    Cook.mix('Bowl ' + str(i), mix)

data2 = dict(van=2, choc=1)
Cook.update(data2)
y1 = Cook.posterior()

plt.plot(x, y, label='Prior')
plt.plot(x, y1.values(), label='Posterior')
plt.xlabel('Bowl Number')
plt.legend()
plt.show()
print(Cook.get_max())

cdf = Cook.credible_interval()

#%%


