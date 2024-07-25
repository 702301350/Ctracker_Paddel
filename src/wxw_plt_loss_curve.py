import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt(
    '/home/titanv/Wangxiangwei/CTracker_annotate/Loss_curve/loss_values.txt',
    delimiter=',', skiprows=1)
Total_loss = data[:, 0]
Cls_loss = data[:, 1]
Reg_loss = data[:, 2]
Reid_loss = data[:, 3]
plt.figure(figsize=(8, 6))
x_ticks = np.arange(0, 1301, 100)
x_labels = [str(i) for i in x_ticks]
plt.xticks(x_ticks, x_labels)
plt.plot(range(len(Total_loss)), Total_loss, label='Total Loss', color='red')
plt.legend(loc='upper right')
plt.xlabel('Iterations (/100)')
plt.ylabel('Value')
plt.title('Total Loss Curve')
plt.xlim(0, 1300)
plt.savefig(
    '/home/titanv/Wangxiangwei/CTracker_annotate/Training_Loss_Curve/total_loss_curve.png'
    )
plt.close()
plt.figure(figsize=(8, 6))
x_ticks = np.arange(0, 1301, 100)
x_labels = [str(i) for i in x_ticks]
plt.xticks(x_ticks, x_labels)
plt.plot(range(len(Cls_loss)), Cls_loss, label='Classification Loss', color
    ='green')
plt.legend(loc='upper right')
plt.xlabel('Iterations (/100)')
plt.ylabel('Value')
plt.title('Classification Loss Curve')
plt.xlim(0, 1300)
plt.savefig(
    '/home/titanv/Wangxiangwei/CTracker_annotate/Training_Loss_Curve/classification_loss_curve.png'
    )
plt.close()
plt.figure(figsize=(8, 6))
x_ticks = np.arange(0, 1301, 100)
x_labels = [str(i) for i in x_ticks]
plt.xticks(x_ticks, x_labels)
plt.plot(range(len(Reg_loss)), Reg_loss, label='Regression Loss', color='blue')
plt.legend(loc='upper right')
plt.xlabel('Iterations (/100)')
plt.ylabel('Value')
plt.title('Regression Loss Curve')
plt.xlim(0, 1300)
plt.savefig(
    '/home/titanv/Wangxiangwei/CTracker_annotate/Training_Loss_Curve/regression_loss_curve.png'
    )
plt.close()
plt.figure(figsize=(8, 6))
x_ticks = np.arange(0, 1301, 100)
x_labels = [str(i) for i in x_ticks]
plt.xticks(x_ticks, x_labels)
plt.plot(range(len(Reid_loss)), Reid_loss, label='Regression Loss', color=
    'blue')
plt.legend(loc='upper right')
plt.xlabel('Iterations (/100)')
plt.ylabel('Value')
plt.title('Regression Loss Curve')
plt.xlim(0, 1300)
plt.savefig(
    '/home/titanv/Wangxiangwei/CTracker_annotate/Training_Loss_Curve/regression_loss_curve.png'
    )
plt.close()
plt.figure(figsize=(8, 6))
x_ticks = np.arange(0, 1301, 100)
x_labels = [str(i) for i in x_ticks]
plt.xticks(x_ticks, x_labels)
plt.plot(range(len(Total_loss)), Total_loss, label='Total Loss', color='red')
plt.plot(range(len(Cls_loss)), Cls_loss, label='Classification Loss', color
    ='green')
plt.plot(range(len(Reg_loss)), Reg_loss, label='Regression Loss', color='blue')
plt.legend(loc='upper right')
plt.xlabel('Iterations (/100)')
plt.ylabel('Value')
plt.title('Combined Loss Curve')
plt.xlim(0, 1300)
plt.savefig(
    '/home/titanv/Wangxiangwei/CTracker_annotate/Training_Loss_Curve/combined_loss_curve.png'
    )
plt.close()
