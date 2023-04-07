from selfsuper_loader import get_self_train_loaders

# Seems to work

def test():
    train_loader, test_loader = get_self_train_loaders(
        path='../../datasets/split_all',
        batch_size=1
    )

    for board, perm in train_loader:
        print(board[0][0])
        print(perm)
        break

if __name__ == '__main__':
    test()
