import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import csv

def code_runner():
    all_res = []
    days =  int(day_textbox.get("1.0", "end-1c"))
    num_gpu =  int(gpunum_textbox.get("1.0", "end-1c"))
    max_layer =  int(layer_textbox.get("1.0", "end-1c"))
    min_layer = int(layerm_textbox.get("1.0", "end-1c"))
    max_mfu =  float(mfu_textbox.get("1.0", "end-1c"))/100
    gputype=  int(gpu_textbox.get("1.0", "end-1c"))
    flops = gputype*1e12*86400*max_mfu*num_gpu*days
    gamma =  float(pre_textbox.get("1.0", "end-1c"))
    min_mmlu =  float(mmlu_textbox.get("1.0", "end-1c"))
    max_size = float(size_textbox.get("1.0", "end-1c"))
    mim_size = float(sizem_textbox.get("1.0", "end-1c")) 
    max_token = float(token_textbox.get("1.0", "end-1c"))
    for ffn_s in np.arange(0,100,2):
        progress_var.set(ffn_s+2) 
        root.update()
        for c1,hidden in enumerate(np.arange(16384,1024,-1024)):
            for c2,n_layer in enumerate(np.arange(min_layer,max_layer)):
                for token_T in np.arange(1,max_token,0.5):

                    hidden = float(hidden)
                    ffn = hidden + 1024 * ffn_s
                    if ffn%4096!=0:
                        continue

                    if hidden%1024!=0:
                        continue

                    if n_layer%8 in [0,1,2,3,4,5]:
                        continue
                    rawtoken_T = float(token_T)
                    size = ((hidden * 128 * 8 * 2 + 2*hidden**2 + 3*hidden*ffn)*n_layer + hidden*150000*2)
                    token_T = min(rawtoken_T, size/1e9)
                    if 6*size*1e12*rawtoken_T>flops or size>max_size*1e9  or size<mim_size*1e9:
                        continue
                    unstable_discount = np.exp(-((10/ffn+20/hidden)*(gamma*n_layer))**2)
                    res = np.sum(np.array([13.95018192,  0.23072027, -0.48523402,  5.39801689]) * np.log(unstable_discount*np.array([n_layer,hidden,ffn,token_T])))+9.195410648934768
                    if res<min_mmlu:
                        continue
                    if res<25:
                        res = 25
                    if res>90:
                        res = 90+np.tanh(0.1*res-9)*10
                    all_res.append([int(hidden),int(ffn), n_layer,round(size/1e9,2),rawtoken_T,round(6*size*1e12*token_T/flops,4),round(res,2)])
    
    for item in tree.get_children():
        tree.delete(item)
        
    if int(var.get())==1:
        for line in sorted(all_res,key=lambda x:x[-1],reverse=True)[:3000]:
            item = [line[2],line[0],line[1],line[3],line[4],line[-1],line[-2]]    
            tree.insert("", tk.END, values=item)
            
    if int(var.get())==2:
        for line in sorted(all_res,key=lambda x:x[2])[:3000]:
            item = [line[2],line[0],line[1],line[3],line[4],line[-1],line[-2]]    
            tree.insert("", tk.END, values=item)
            
    if int(var.get())==3:
        for line in sorted(all_res,key=lambda x:x[-1]**2/np.log(1+x[2]),reverse=True)[:3000]:
            item = [line[2],line[0],line[1],line[3],line[4],line[-1],line[-2]]    
            tree.insert("", tk.END, values=item)
        

root = tk.Tk()
root.title("Training Plan Generator Based on Performance Law")

api_label = tk.Label(root, text="BF16 TFLOPS:")
api_label.grid(row=0, column=0, padx=5, pady=1, sticky="w")

gpu_textbox = tk.Text(root ,height=1, width=30)
gpu_textbox.insert(tk.END, '376')
gpu_textbox.grid(row=1, column=0, padx=5, pady=1, sticky="w")

api_label = tk.Label(root, text="GPUs:")
api_label.grid(row=0, column=1, padx=5, pady=1, sticky="w")

gpunum_textbox = tk.Text(root ,height=1, width=30)
gpunum_textbox.insert(tk.END, '1024')
gpunum_textbox.grid(row=1, column=1, padx=5, pady=1, sticky="w")


api_label = tk.Label(root, text="MFU（%）:")
api_label.grid(row=0, column=2, padx=5, pady=1, sticky="w")

mfu_textbox = tk.Text(root ,height=1, width=30)
mfu_textbox.insert(tk.END, '40')
mfu_textbox.grid(row=1, column=2, padx=5, pady=1, sticky="w")


api_label = tk.Label(root, text="Training Time（Day）:")
api_label.grid(row=2, column=0, padx=5, pady=1, sticky="w")

day_textbox = tk.Text(root ,height=1, width=30)
day_textbox.insert(tk.END, '30')
day_textbox.grid(row=3, column=0, padx=5, pady=1, sticky="w")


api_label = tk.Label(root, text="Max Layers:")
api_label.grid(row=2, column=1, padx=5, pady=1, sticky="w")

layer_textbox = tk.Text(root ,height=1, width=30)
layer_textbox.insert(tk.END, '100')
layer_textbox.grid(row=3, column=1, padx=5, pady=1, sticky="w")

api_label = tk.Label(root, text="Max Training Tokens（T）:")
api_label.grid(row=2, column=2, padx=5, pady=1, sticky="w")

token_textbox = tk.Text(root ,height=1, width=30)
token_textbox.insert(tk.END, '20')
token_textbox.grid(row=3, column=2, padx=5, pady=1, sticky="w")

api_label = tk.Label(root, text="Min Size（B）:")
api_label.grid(row=4, column=0, padx=5, pady=1, sticky="w")

sizem_textbox = tk.Text(root ,height=1, width=30)
sizem_textbox.insert(tk.END, '10')
sizem_textbox.grid(row=5, column=0, padx=5, pady=1, sticky="w")


api_label = tk.Label(root, text="Max Size（B）:")
api_label.grid(row=4, column=1, padx=5, pady=1, sticky="w")

size_textbox = tk.Text(root ,height=1, width=30)
size_textbox.insert(tk.END, '100')
size_textbox.grid(row=5, column=1, padx=5, pady=1, sticky="w")


api_label = tk.Label(root, text="Precision Loss γ（≥1）:")
api_label.grid(row=4, column=2, padx=5, pady=1, sticky="w")

pre_textbox = tk.Text(root ,height=1, width=30)
pre_textbox.insert(tk.END, '1')
pre_textbox.grid(row=5, column=2, padx=5, pady=1, sticky="w")

api_label = tk.Label(root, text="Min layers:")
api_label.grid(row=6, column=0, padx=5, pady=1, sticky="w")

layerm_textbox = tk.Text(root ,height=1, width=30)
layerm_textbox.insert(tk.END, '20')
layerm_textbox.grid(row=7, column=0, padx=5, pady=1, sticky="w")


api_label = tk.Label(root, text="Min MMLU:")
api_label.grid(row=6, column=1, padx=5, pady=1, sticky="w")

mmlu_textbox = tk.Text(root ,height=1, width=30)
mmlu_textbox.insert(tk.END, '50')
mmlu_textbox.grid(row=7, column=1, padx=5, pady=1, sticky="w")


split_button = tk.Button(root, text="Compute", command=code_runner, width=95,height=1)
split_button.grid(row=8, column=0,columnspan=3, padx=5, pady=5, sticky="w")

progress_var = tk.IntVar()


progressbar = ttk.Progressbar(root, variable=progress_var, maximum=100,length=675)
progressbar.grid(row=9, column=0,columnspan=3, padx=5, pady=5, sticky="w")


var = tk.StringVar(value="1")  

radiobutton1 = tk.Radiobutton(root, text="Performance First", variable=var,width=15, value="1" )
radiobutton1.grid(row=10, column=0, padx=5, pady=1, sticky="w")

radiobutton2 = tk.Radiobutton(root, text="Shallow Model First", variable=var,width=15, value="2" )
radiobutton2.grid(row=10, column=1, padx=5, pady=1, sticky="w")

radiobutton4 = tk.Radiobutton(root, text="Balance", variable=var,width=15, value="3" )
radiobutton4.grid(row=10, column=2, padx=5, pady=1, sticky="w")


tree = ttk.Treeview(root, columns=("Layer", "Hidden", "FFN",'Para','Tokens','MMLU','Budget'), show='headings')
tree.grid(row=11, column=0,columnspan=3, padx=5, pady=1, sticky="w")

scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.grid(row=11, column=3, sticky='ns')

tree.column("Layer", width=95)
tree.column("Hidden", width=95)
tree.column("FFN", width=95)
tree.column("Para", width=95)
tree.column("Tokens", width=95)
tree.column("MMLU", width=95)
tree.column("Budget", width=95)
tree.heading("Layer", text="Number of Layers")
tree.heading("Hidden", text="Hidden Size")
tree.heading("FFN", text="FFN Dimension")
tree.heading("Para", text="Parameters/B")
tree.heading("Tokens", text="Training Tokens/T")
tree.heading("MMLU", text="MMLU")
tree.heading("Budget", text="Budget Utilization")


api_label = tk.Label(root, text="Output Path")
api_label.grid(row=12, column=0, padx=5, pady=1, sticky="w")

path_textbox = tk.Text(root ,height=1, width=30)
path_textbox.insert(tk.END, 'result.csv')
path_textbox.grid(row=12, column=0, padx=5, pady=10, sticky="w")

def export_to_csv():

    with open(path_textbox.get("1.0", "end-1c"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(tree["columns"])
        for item in tree.get_children():
            writer.writerow(tree.item(item)["values"])

export_button = tk.Button(root, text="Export to CSV", command=export_to_csv)
export_button.grid(row=12, column=1, pady=10, sticky='w')

root.mainloop()
