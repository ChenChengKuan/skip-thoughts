import vocab
import train
import os
import sys
import argparse
sys.path.append("../")
import skipthoughts
os.environ["THEANO_FLAGS"] = "device=cuda2"
SKMODEL = "/media/VSlab3/kuanchen_arxiv/NeuralStoryTeller/"

def main():
    parser = argparse.ArgumentParser(description='Pass target style genre to train decoder')
    parser.add_argument('-s', '--style_genre', help='the name of style corpus', required='True', default='localhost')
    flag = parser.parse_args()

    style_corpus_path = "/media/VSlab3/kuanchen_arxiv/artistic_style_corpora/{}".format(flag.style_genre)
    style_genre = flag.style_genre.split(".")[0]

    X = []
    with open(style_corpus_path, 'r') as handle:
        for line in handle.readlines():
            X.append(line.strip())
    C = X
    if not os.path.isfile("./vocab_save/{}.pkl".format(style_genre)):
        print "Get vocabulary..."
        worddict, wordcount = vocab.build_dictionary(X)
        vocab.save_dictionary(worddict=worddict, wordcount=wordcount, loc="vocab_save/{}.pkl".format(style_genre))
    else:
        pass
    savepath = "./logs_{}".format(style_genre)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    skmodel = skipthoughts.load_model() 
    train.trainer(X, C, skmodel, dictionary="vocab_save/{}.pkl".format(style_genre), savepath=savepath ,saveto="model.npz")

if __name__ == '__main__':
    main()
