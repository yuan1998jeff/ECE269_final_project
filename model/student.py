import torch.nn as nn
import torchvision

class Student_Network(nn.Module):
    def __init__(self, num_classes=35, criterion=nn.CrossEntropyLoss(), down_sample=8):
        super(Student_Network, self).__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)
        self.criterion = criterion
        self.down_sample = down_sample
    def forward(self, input):
        _,_,H,W = input.size()
        out = self.model(input)
        logits = nn.functional.interpolate(out['out'],size=(224//self.down_sample,448//self.down_sample))
        return logits
    def _loss(self, input, target):
        logits = self.forward(input)
        loss = self.criterion(logits, target)
        return loss
    '''
    def parameters(self):
        return self.model.parameters()
    def named_parameters(self):
        return self.model.named_parameters()
    def state_dict(self):
        return self.model.state_dict()
        '''
    def new(self):
        new_model = Student_Network(self.num_classes, self.criterion, self.down_sample)
        return new_model.cuda()
    '''
    def load_state_dict(self, model_dict):
        self.model.load_state_dict(model_dict)
        '''