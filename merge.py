import os
import glob

batch_size = 100
cnn_dir = '/home/mjc/datasets/CNN_DailyMail/cnn/stories_idx/'
cnn_mer_dir = '/home/mjc/datasets/CNN_DailyMail/cnn/stories_idx_merged_%d/' % batch_size

file_list = [os.path.join(cnn_dir,file) for file in os.listdir(cnn_dir)]
files_read = 0
cnt = 1
while (files_read<len(file_list)):
	batch = file_list[files_read:min(files_read+batch_size,len(file_list))]
	out = []
	for file_name in batch:
		with open(file_name,'r') as f:
			text = f.read()
		text = text.strip()
		out.append(text)
	out = '\n'.join(out)
	with open(cnn_mer_dir+'train_%d.txt' % (cnt),'w') as f:
		f.write(out)
	cnt+=1
	files_read+=len(batch)
	print("%d files merged..." % (files_read))
