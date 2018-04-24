import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO

class Logger:
	def __init__(self,
				 log_interval=50,
				 validation_interval=300,
				 trainer=None):
		self.log_interval = log_interval
		self.validation_interval = validation_interval
		self.accumulated_loss = 0

	def log(self,current_step,current_loss):
		self.accumulated_loss  += current_loss
		if current_step % self.log_interval == 0:
			self.log_loss(current_step)
			self.accumulated_loss = 0

		if current_step %self.validation_interval == 0:
			self.validate(current_step)

	def log_loss(self,current_step):
		avg_loss = self.accumulated_loss / self.log_interval
		print 'loss at step '+str(current_step)+': '+str(avg_loss)

	def validate(self,current_step):
		avg_loss = self.trainer.validate()
		print 'validation loss: '+str(avg_loss)
		#print 'validation accuracy: '+str(avg_accuracy*100)+'%'


