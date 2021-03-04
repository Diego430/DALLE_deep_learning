
class ImageCaptions :
	"""
	an iterator for fetching data during training
	"""
	def __init__(self, data, batchsize = 4) :
		self.data = data
		self.len = len(data)
		self.index = 0
		self.end = False
		self.batchsize = batchsize

	def __iter__(self) :
		return self

	def __next__(self) :
		if self.end :
			self.index = 0
			raise StopIteration
		image_data = []
		caption_data = []
		for i in range(0, self.batchsize) :
			image_data.append(self.data[self.index][0])
			caption_tokens = [0] * 256  # fill to match text_seq_len
			caption_tokens_data = self.data[self.index][1]
			caption_tokens[:len(caption_tokens_data)] = caption_tokens_data
			caption_data.append(caption_tokens)
			self.index += 1
			if self.index == self.len :
				self.end = True
				break
		return image_data, caption_data
