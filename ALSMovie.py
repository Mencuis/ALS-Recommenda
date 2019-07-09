import numpy as np
import random


class MyALS:
    def __init__(self, user_ids, item_ids, ratings, rank, iterations=5, lambda_=0.01, blocks=-1, seed=None):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.rank = rank
        self.iterations = iterations
        self.lambda_ = lambda_
        self.blocks = blocks
        self.seed = seed
        self.user_matrix = None
        self.item_matrix = None
        self.rmse = None
        self.ratings_size = ratings.shape
        self.user_dict = dict(zip(user_ids, range(len(user_ids))))
        self.item_dict = dict(zip(item_ids, range(len(item_ids))))

    def _preprocess(self):
        user_n = len(self.user_ids)
        item_n = len(self.item_ids)
        if self.ratings_size != (user_n, item_n):
            print("matrix ratings must be suitable with user_ids and iten_ids")
            raise IndexError

    def _random_matrix(self):
        np.random.seed(self.seed)
        self.user_matrix = np.random.rand(self.ratings_size[0], self.rank) 

    def _get_rmse(self):
        """" rmse = sqrt(sum[(ratings - premat)^2]/N) """
        predict_matrix = np.matmul(self.user_matrix, self.item_matrix)
        de = np.array(self.ratings - predict_matrix)
        self.rmse = (sum(sum(de ** 2)) / (self.ratings_size[0] * self.ratings_size[1])) ** 0.5

    def _get_item_matrix(self):
        """ Y = (X^T*X + lambda*I)^-1*X^T*ratings """
        self.item_matrix = np.matmul(
            np.matmul((np.linalg.pinv(np.matmul(self.user_matrix.T, self.user_matrix) + self.lambda_)),
                      self.user_matrix.T), ratings)

    def _get_user_matrix(self):
        """ X = ((Y*Y^T + lambda*I)^-1*Y*ratings^T)^T """
        self.user_matrix = np.matmul(
            np.matmul((np.linalg.pinv(np.matmul(self.item_matrix, self.item_matrix.T) + self.lambda_)),
                      self.item_matrix), ratings.T).T

    def learn_para(self, rankrange, iterationrange, lambdarange):
        self._preprocess()
        para = dict()
        for self.rank in range(rankrange[0], rankrange[1], rankrange[2]):
            for self.iterations in range(iterationrange[0], iterationrange[1], iterationrange[2]):
                for self.lambda_ in np.arange(lambdarange[0], lambdarange[1], lambdarange[2]):
                    print("para <rank, iteration, lambda>", self.rank, self.iterations, self.lambda_)
                    self._random_matrix()
                    self._get_item_matrix()
                    self._get_user_matrix()
                    self._get_rmse()
                    firstrmse = self.rmse
                    for k in range(self.iterations-1):
                        self._get_item_matrix()
                        self._get_user_matrix()
                        self._get_rmse()
                    if self.rmse < firstrmse:
                        para[self.rank, self.iterations, self.lambda_] = self.rmse
                        print("converge, rmse: ", self.rmse)
        return  para


    def fit(self):
        self._preprocess()
        self._random_matrix()
        for k in range(self.iterations):
            self._get_item_matrix()
            self._get_user_matrix()
            self._get_rmse()
            print("Iterations: {0}, RMSE: {1:.6}".format(k + 1, self.rmse))

    def predict(self, user_id, n_items=10):
        if type(user_id) is not list:
            user_id = [user_id]
        k = []
        predict_user_matrix = np.zeros((len(user_id), self.rank))
        for m in range(len(user_id)):
            k.append(self.user_ids.index(user_id[m]))
            predict_user_matrix[m] = self.user_matrix[k[m]]
        scores_matrix = np.matmul(predict_user_matrix, self.item_matrix)
        scores_dict = dict()
        # recommend = dict()
        t = 0
        for id1 in user_id:
            scores_dict[id1] = dict(zip(self.item_ids, scores_matrix.tolist()[t]))
            for item in item_ids:
                if self.ratings[self.user_dict[id1], self.item_dict[item]] != 0:
                    del scores_dict[id1][item]
            t += 1
        recc = dict()
        for id1 in user_id:
            recc[id1] = sorted(scores_dict[id1].items(), key=lambda x: x[1], reverse=True)[:n_items]
        return recc


if __name__ == "__main__":
    print("test ALS...")
    """
    user_ids = [1, 22, 333, 4444]
    item_ids = [111, 222, 333, 444, 555, 666, 777]
    ratings = np.matrix([[0, 4, 3, 5, 0, 0, 1],
                        [2, 0, 5, 1, 0, 0, 3],
                        [5, 3, 2, 1, 3, 5, 0],
                        [1, 2, 3, 4, 5, 3, 2]])
    #"""
    user_ids = random.sample(range(1, 1000), 300)
    item_ids = random.sample(range(1, 10000), 2000)
    ratings = np.random.randint(0, 5, (300, 2000))
    #"""
    
    model = MyALS(user_ids, item_ids, ratings, rank=10, iterations=5, lambda_=0.1)
    model.fit()
    rec = model.predict(user_ids[:3], 3)
    print(rec)
    #model = MyALS(user_ids, item_ids, ratings, rank=1, seed=1)
    #para = model.learn_para([1, 100, 10], [5, 20, 5], [0.01, 0.5, 0.04])
    #print(para)
