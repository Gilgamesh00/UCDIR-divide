from torchvision import transforms
from .randaugment import RandAugmentMC
from .autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy


def office_home_train(mean, std):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def office_home_test(mean, std):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


class OHMultiView(object):
    '''
        Multi-view transformation for office home
    '''
    def __init__(self, mean, std, views='ww', aug='auto'):
        # all() 是一个 Python 内置函数，其作用是判断给定可迭代对象中所有元素是否都为 True。如果是，则返回 True，否则返回 False
        assert all(v in 'wst' for v in views)
        assert aug in ['rand', 'auto']
        self.views = views

        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),      # weak aug
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10) if aug == 'rand' else ImageNetPolicy(),    # strong aug: fixmatch
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        ret_views = []
        for v in self.views:
            if v == 'w':
                ret_views.append(self.weak(x))
            elif v == 's':
                ret_views.append(self.strong(x))
            else:  # 't'
                ret_views.append(self.test(x))
        if len(ret_views) > 1:
            return ret_views
        else:
            return ret_views[0]
