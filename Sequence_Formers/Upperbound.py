import argparse
import os.path
import torch


class Upperbound(object):
    def __init__(self, args):
        super(Upperbound, self).__init__()

        self.n_tasks = args.n_tasks
        self.i = args.i
        self.train_file = args.train_file
        self.test_file = args.test_file
        self.o = os.path.join(self.i, self.train_file).replace('training.pt','upperbound_'+str(self.n_tasks)+'.pt')
        self.o = os.path.join(args.o, 'upperbound_'+str(self.n_tasks)+'.pt')


    def formating_data(self):

        # we create a data loader with first only zero then zero and one .... util all digits are in the list

        assert os.path.isfile(os.path.join(self.i, self.train_file))
        assert os.path.isfile(os.path.join(self.i, self.test_file))

        tasks_tr = []
        tasks_te = []

        x_tr, y_tr = torch.load(os.path.join(self.i, self.train_file))
        x_te, y_te = torch.load(os.path.join(self.i, self.test_file))

        x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
        x_te = x_te.float().view(x_te.size(0), -1) / 255.0
        y_tr = y_tr.view(-1).long()
        y_te = y_te.view(-1).long()

        cpt = int(10 / self.n_tasks)

        for t in range(self.n_tasks):
            c1 = 0
            c2 = (t + 1) * cpt
            i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
            i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
            tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
            tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])

        torch.save([tasks_tr, tasks_te], self.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i', default='raw/cifar100.pt', help='input directory')
    parser.add_argument('--o', default='cifar100.pt', help='output file')
    parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    DataFormater = Upperbound()
    DataFormater.formating_data(args)
