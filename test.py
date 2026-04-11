import torch

# a = torch.rand(10,5)
# print(" a : ", a)
# print("dim of a : ", a.dim())
# #   
# # print("sorted a : ", torch.sort(a,descending=True))
# #   
# values, indices = torch.topk(a,k=3,largest=True,sorted=True)
# # print("values : ", values)
# print("indices : ", indices)

    
# # after_selected = torch.index_select(a,0, indices)
# # print("after selected : ", after_selected)

# refer_select = torch.zeros(10,3)
# for i in range(10):
#     refer_select[i,:] = a[i,indices[i,:]]
    
# print("after select : ", refer_select) 

# a = torch.rand(3,4,2,3)
# print("rand a : ", a)
# a_view = a.view(3,2,12)

# b_permute = a.permute(0,2,1,3).contiguous()
# b_permute_view = b_permute.view(3,2,12)
# print("a view : ", a_view)
# print("b permute view : ", b_permute_view)

a = (50/7.5)
b = (50/7)
c = 100/14.5

d = (a+b)/2
print(a,b,c,d)