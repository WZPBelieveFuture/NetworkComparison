from gensim.models import Word2Vec
from RandoWalk import RandomWalker
class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):
        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(graph, p=p, q=q, )
        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)
    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        walk=self.sentences
        new_walk=[]
        for i in walk:
            new_list=[str(x) for x in i]
            new_walk.append(new_list)
        kwargs["sentences"] = new_walk
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter
        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")
        self.w2v_model = model
        return model
    def get_embeddings(self,Number):
        if self.w2v_model is None:
            print("model not train")
            return {}
        self._embeddings = {}
        # for word in self.graph.nodes():
        #     self._embeddings[word] = self.w2v_model.wv[str(word)]
        sum_embedding=0
        empty_list=[]
        for i in range(Number):
            if str(i) in self.w2v_model.wv:
                self._embeddings[i]=self.w2v_model.wv.word_vec(str(i))
                sum_embedding+=self._embeddings[i]
            else:
                empty_list.append(i)
        mean_embedding=sum_embedding / (Number-len(empty_list))
        for i in empty_list:
            self._embeddings[i]=mean_embedding
        return self._embeddings
