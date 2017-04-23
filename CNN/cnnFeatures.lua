-- CNN part of network
-- Run volumes through CNN and concatenate features

require 'torch';   -- torch
require 'image' ;  -- for image transforms
require 'nn' ;     -- provides all sorts of trainable modules/layers
require 'cudnn';
require 'rnn';
require 'cunn';

py = require('fb.python')

-- Load number of slices and timepoints
py.exec([=[
import numpy as np
from openpyxl import load_workbook
wb = load_workbook(filename = '../data.xlsx')
sh = wb['Sheet']
nSlices = np.zeros(500)
nTimePoints = np.zeros(500)
skip = np.zeros(500)
for i in range(500):
	nSlices[i] = sh['D'+str(i+2)].value
	nTimePoints[i] = sh['H'+str(i+2)].value
	if nTimePoints[i]>30 and nTimePoints[i] %30==0:
		nTimePoints[i] = 30
	skip[i] = sh['I'+str(i+2)].value
]=])

nSlices = py.eval('nSlices')
nTimePoints = py.eval('nTimePoints')
skip = py.eval('skip')

--Loading images
print('==> Loading images..')

py.exec([=[
import numpy as np
main = '/windows/athira/reslicedData/'
start = 201
last = 300
data = np.zeros(shape = ((last-start+1),30,11,256,192))
for pnum in range(start,last+1):
	print('--- Loading patient no.'+str(pnum)+'..')
	for tp in range(1,31):
		fname = main+'p'+str(pnum)+'_tp'+str(tp)+'.npy'
		temp = np.load(fname)
		data[pnum-start,tp-1,:,:,:] = temp.reshape([11,256,192])
]=])


-- Load model
print('==> Loading model..')
model = torch.load('results/model.net')
model:evaluate()


data = py.eval('data')

mean = torch.load('dataMean.t7')
std = torch.load('dataStd.t7')
data = data:add(-mean)
data = data:div(std)


--Changes: python script above, start and last

start = 201
last = 300
k = 0
des_nSlices = 11
-- a = np.array(nSlices[100:122])
-- b = np.array(nTimePoints[100:122])
-- k = k + int(sum(a*b))
-- print k



inputSize = 1000

for pnum = start,last do
	if skip[pnum]==0 then
		ns = 11
		nt = 30
		if nTimePoints[pnum]%30~=0 then
			print('No. of timepoints is not 30!!')
		end

		print('Running Patient no.' .. pnum..'..')
		
		sampFeat = {}
		for t = 1,nt do
			
			for sl = 1,ns do
				img = torch.Tensor(1,256,192)
				img[{1}] = data[{pnum-start+1,t,sl,{},{}}]
				out = model:forward(img:cuda())
				-- print(out)
				feats = model:get(7).output
				-- print(feats:size())
				nx = math.floor(out[1]*feats:size(2))
				ny = math.floor(out[2]*feats:size(3))
				s = math.floor(0.4 * feats:size(2))
				-- print(nx..' '..ny..' '..s)

				q = 0
				if q == 1 and pnum==30 then
					results = {img,out:float()}
					print('DEBUG: Writing img..')
					py.exec([=[	
import numpy as np 
import nibabel as nib
desSize = (256,192)
img = res[0]
out = res[1]
img = img.reshape(desSize[0],desSize[1])
img2 = img/np.max(img)
box = out
tbox = np.zeros(3)
tbox[0] = box[0]*desSize[0]
tbox[1] = box[1]*desSize[1]
tbox[2] = box[2]*desSize[0]
(x1,y1,x2,y2) = (tbox[0]-tbox[2]/2,tbox[1]-tbox[2]/2,tbox[0]+tbox[2]/2,tbox[1]+tbox[2]/2)

img2[y1,x1:x2] = 1
img2[y2,x1:x2] = 1
img2[y1:y2,x1] = 1
img2[y1:y2,x2] = 1
img3 = nib.Nifti1Image(img2, np.eye(4))
img3.get_data_dtype() == np.dtype(np.float32)
img3.header.get_xyzt_units()
img3.to_filename('img3.nii')
	]=],{res = results})
					print('DEBUG: Writing img done.')
				end

				-- checks
				x_min = nx-s/2
				x_max = nx+s/2
				y_min = ny-s/2
				y_max = ny+s/2
				if x_min <1 then
					temp = 1-x_min
					x_min = 1
					x_max = x_max+temp
				end
				if y_min <1 then
					temp = 1-y_min
					y_min = 1
					y_max = y_max+temp
				end
				if x_max > feats:size(2) then
					temp = x_max-feats:size(2)
					x_max = feats:size(2)
					x_min = x_min-temp
				end
				if y_max > feats:size(3) then
					temp = y_max-feats:size(3)
					y_max = feats:size(3)
					y_min = y_min-temp
				end

				featMat = feats[{{},{x_min,x_max},{y_min,y_max}}]
				featVec = featMat:reshape(featMat:size(1)*featMat:size(2)*featMat:size(3))
				if sl == 1 then
					tpVec = featVec
					-- print('first')
					-- print(tpVec:size())
				else
					tpVec = torch.cat(tpVec,featVec)
					-- print(tpVec:size())
				end

			end
			table.insert(sampFeat,tpVec)
			-- k = k+ns
		end

		torch.save('../LSTMdata/pnum_'..pnum..'.t7',sampFeat)
	end

end


