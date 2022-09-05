#coding:utf8
import visdom
import time
import numpy as np
class Visualizer(object):
	'''
	self.text('hello visdom')
	self.histogram(t.randn(1000))
	self.line(t.arange(0, 10),t.arange(1, 11))
	'''

	def __init__(self, env='default', **kwargs):
		self.vis = visdom.Visdom(env=env, **kwargs)

		self.index = {}
		self.log_text = ''
	def reinit(self, env='default', **kwargs):
		self.vis = visdom.Visdom(env=env, **kwargs)
		return self

	def plot_many(self, d):
		'''
		@params d: dict (name, value) i.e. ('loss', 0.11)
		'''
		for k, v in d.iteritems():
			self.plot(k, v)

	def img_many(self, d):
		for k, v in d.iteritems():
			self.img(k, v)

	def plot(self, name, y, **kwargs):
		'''
		self.plot('loss', 1.00)
		'''
		x = self.index.get(name, 0)
		self.vis.line(Y=np.array([y]), X=np.array([x]),
					 win = unicode(name),
					 opts = dict(title=name),
					 update = None if x == 0 else 'append',
					 **kwargs
					  )
		self.index[name] = x + 1
	def plot_many_stack(self, d, win_name):
		name=list(d.keys())
		name_total=" ".join(name)
		x = self.index.get(name_total, 0)
		val=list(d.values())
		if len(val) == 1:
			y = np.array(val)
		else:
			y = np.array(val).reshape(-1, len(val))
		#print(x)
		self.vis.line(Y=y,X=np.ones(y.shape)*x,
					win=str(win_name),#unicode
					opts=dict(legend=name,
						title=win_name),
					update=None if x == 0 else 'append'
					)
		self.index[name_total] = x + 1     
	def img(self, name, img_, **kwargs):
		'''
		self.img('input_img', t.Tensor(64, 64))
		self.img('input_imgs', t.Tensor(3, 64, 64))
		self.img('input_imgs', t.Tensor(100, 1, 64, 64))
		self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
		'''
		self.vis.images(img_.cpu().numpy(),
					  win=unicode(name),
					  opts=dict(title=name),
					  **kwargs
					  )

	def log(self, info, win='log_text'):
		'''
		self.log({'loss':1, 'lr':0.0001})
		'''
		self.log_text += ('[{time}] {info} <br>'.format(
						   time=time.strftime('%m%d_%H%M%S'),\
						   info=info)) 
		self.vis.text(self.log_text, win)
	def __getattr__(self, name):
		return getattr(self.vis, name)

if __name__ == '__main__':
	  
	from torchnet import meter
	''' 
	vis = Visualizer(env='my_wind')
	loss_meter = meter.AverageValueMeter()
	for epoch in range(10):
		loss_meter.reset()
		model.train()
		for ii,(data,label)in enumerate(trainloader):     
			...
			out=model(input)
			loss=...
			loss_meter.add(loss.data[0])
		#loss
		vis.plot_many_stack({'train_loss': loss_meter.value()[0]})    
	'''
	vis = Visualizer(env='loss')
	train_loss = meter.AverageValueMeter()
	val_loss = meter.AverageValueMeter()
	train_acc = meter.AverageValueMeter()
	val_acc = meter.AverageValueMeter()
	for epoch in range(1, 10):
		train_loss.reset()
		val_loss.reset()
		time.sleep(1)
		train_loss.add(np.exp(epoch+1))
		val_loss.add(np.log(epoch))#print(loss_meter.value())
		train_acc.add(np.exp(epoch+1))
		val_acc.add(epoch+2)
		vis.plot_many_stack({'train_loss':train_loss.value()[0],'test_loss':val_loss.value()[0]},win_name = "resnet18/loss")
		#vis.plot_many_stack({'train_acc':train_acc.value()[0]})
		#time.sleep(3)
		#vis.plot_many_stack({'train_loss': loss_meter.value()[0]})
		