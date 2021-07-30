import matplotlib.pyplot as plt

class plotgraph:
    def __init__(self, loss_list, valloss_list, valacc_list, path, description=""):
        self.path = path
        self.loss_list = loss_list
        self.valloss_list = valloss_list
        self.valacc_list = valacc_list

        # print(min(self.loss_list))
        # print(self.loss_list.index(min(self.loss_list)))
        # print(min(self.valloss_list))
        # print(self.valloss_list.index(min(self.valloss_list)))
        # print(max(self.valacc_list))
        # print(self.valacc_list.index(max(self.valacc_list)))

        print("Saving loss, accuracy graph... ...")
        plt.figure()
        plt.title('model loss')
        plt.plot(self.loss_list)
        plt.plot(self.valloss_list)
        plt.text(self.loss_list.index(min(self.loss_list)),min(self.loss_list),
                str(min(self.loss_list)),
                color='b',
                horizontalalignment='center',
                verticalalignment='bottom')
        plt.text(self.valloss_list.index(min(self.valloss_list)),min(self.valloss_list),
                str(min(self.valloss_list)),
                color='r',
                horizontalalignment='center',
                verticalalignment='bottom')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(self.path + '/loss_' + description + '.png')
        plt.close("all")

        plt.figure()   
        plt.title('validation accuracy')
        plt.plot(self.valacc_list)
        plt.text(self.valacc_list.index(max(self.valacc_list)),max(self.valacc_list),
                str(max(self.valacc_list)),
                color='r',
                horizontalalignment='center',
                verticalalignment='bottom')
        # plt.ylim([0, 100])     # Y축의 범위: [ymin, ymax]
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(['accuracy'], loc='lower right')
        plt.savefig(self.path + '/accuracy_' + description + '.png')
        plt.close("all")
        print("Done.")

