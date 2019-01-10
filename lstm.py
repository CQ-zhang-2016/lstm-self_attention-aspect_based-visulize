import torch
import torch.utils.data
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import time, datetime


# vocab_size (plus one for <unk>)
# weight np.array
# labels number of class
class SentimentLstm(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels):
        super(SentimentLstm, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.embedding.weight= torch.nn.Parameter(torch.FloatTensor(weight))
        self.embedding.weight.requires_grad = False
        
        self.encoder = torch.nn.LSTM(input_size=embed_size, 
                                     hidden_size=self.num_hiddens,
                                     num_layers=self.num_layers,
                                     bidirectional=self.bidirectional,
                                     dropout=0.0)
        
        if self.bidirectional:
            self.decoder = torch.nn.Linear(num_hiddens * 4, 32)
        else:
            self.decoder = torch.nn.Linear(num_hiddens * 2, 32)
        self.decoder_ = torch.nn.Linear(32, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        outputs = torch.nn.functional.relu(outputs)
        outputs = self.decoder_(outputs)
        outputs = torch.nn.functional.softmax(outputs)
        return outputs

if __name__=="__main__":
    # some settings
    f = open(current_path + '\\embedding_matrix3.pickle','rb')
    model = pickle.load(f)
    vocab_size = len(model.wv.vocab.keys())
    #embedding_dim = model.wv.vector_size
    pretrained_weights = np.array(model.wv.vectors)
    pretrained_weights = np.concatenate(
        [pretrained_weights, np.zeros((1, 300), dtype=np.float32)],
        axis=0) #在最后面加一个零向量代表<unk>

    DATA_PATH = current_path + '\\data\\data1.npy'
    num_epochs = 10
    embed_size = 300
    #num_hiddens = 20 ####
    #num_layers = 4
    num_hiddens = 200 ####可以改
    num_layers = 3###可以改
    bidirectional = True
    batch_size = 1 ####
    labels = 5 # 5分类
    lr = 1e-4
    weight_decay = 1e-5

    device = torch.device('cuda:0')
    '''cuda = True
    if cuda:
        torch.cuda.set_device(device)'''

    net = SentimentLstm(vocab_size=(vocab_size+1), embed_size=embed_size,
                        num_hiddens=num_hiddens, num_layers=num_layers,
                        bidirectional=bidirectional, weight=pretrained_weights,
                        labels=labels)
    net.to(device)
    #loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.MSELoss(reduce=True, size_average=True)
    weight_p, bias_p = [],[]
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    # 这里的model中每个参数的名字都是系统自动命名的，只要是权值都是带有weight，偏置都带有bias，
    # 因此可以通过名字判断属性，这个和tensorflow不同，tensorflow是可以用户自己定义名字的，当然也会系统自己定义。

    optimizer = torch.optim.Adam([
              {'params': weight_p, 'weight_decay':1e-5},
              {'params': bias_p, 'weight_decay':0}
              ], lr=lr)


    # load the dataset
    # 95% train, 5% validation
    data = np.load(DATA_PATH)
    data = np.array(data, dtype=np.int32)
    reviews = data[:, :-1]
    scores = data[:, -1]
    scores = scores - 1
    #scores = np_utils.to_categorical(scores)
    num_samples = 2360000
    scores_onehot = np.zeros((num_samples, 5), dtype=np.float32)
    scores_onehot[np.arange(num_samples), scores] = 1.0
    scores = scores_onehot

    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index = index[:int(0.95 * num_samples)]
    test_index = index[int(0.95 * num_samples):]

    train_features = torch.tensor(reviews[train_index])
    train_labels = torch.tensor(scores[train_index])
    test_features = torch.tensor(reviews[test_index])
    test_labels = torch.tensor(scores[test_index])

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    test_set = torch.utils.data.TensorDataset(test_features, test_labels)

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=True)

    '''for epoch in range(num_epochs):
        #print('#' * 80)
        #print('EPOCH: ', epoch)
        #print('#' * 80)
        start = time.time()
        train_loss, test_losses = 0, 0
        train_acc, test_acc = 0, 0
        n, m = 0, 0
        for feature, label in train_iter:
            n += 1
            #net.train()
            net.zero_grad()
            feature = torch.autograd.Variable(feature.cuda()).long()
            label = torch.autograd.Variable(label.cuda()).float()
            score = net(feature)
            loss = loss_function(score, label)
            loss.backward()
            #scheduler.step()
            optimizer.step()
            acc = accuracy_score(torch.argmax(score.cpu().data,
                                              dim=1), torch.argmax(label.cpu().data, dim=1))
            train_acc += acc
            train_loss += loss
            print(datetime.datetime.now(), ' batch: ', n, 'train_acc: ', acc, 'train_loss: ', loss.cpu().data)
        with torch.no_grad():
            for test_feature, test_label in test_iter:
                m += 1
                #net.eval()
                test_feature = test_feature.cuda().long()
                test_label = test_label.cuda().float()
                test_score = net(test_feature)
                test_loss = loss_function(test_score, test_label)
                acc = accuracy_score(torch.argmax(test_score.cpu().data,
                                                  dim=1), torch.argmax(test_label.cpu().data, dim=1))
                test_acc += acc
                test_losses += test_loss

                print(datetime.datetime.now(), ' batch: ', m, 'test_acc: ', acc, 'test_loss: ', test_loss.cpu().data)

        end = time.time()
        runtime = end - start
        print()
        print('$' * 80)
        print('epoch: %d, train loss: %.4f, train acc: %.2f, test loss: %.4f, test acc: %.2f, time: %.2f' %
              (epoch, train_loss / n, train_acc / n, test_losses / m, test_acc / m, runtime))

    torch.save(net.state_dict(),'model_params.pth')
    torch.save(net, 'model.pth')'''

m = torch.load('model.pth')
for test_feature, test_label in test_iter:
    test_feature = test_feature.cuda().long()
    print(test_feature.shape)
    print(m(test_feature))
    print(torch.max(m(test_feature),1)[1])
    exit()
