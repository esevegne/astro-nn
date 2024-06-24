


df = df.drop(df.columns.drop(input_features+output_targets),axis=1)

fig, ax = plt.subplots(len(input_features), 2, figsize=(14,12))
plt.subplots_adjust(hspace=0.5)


for i, col in enumerate(input_features):
    ax[i,0].plot(df[col], label=col)
    ax[i,0].legend(loc='upper right',bbox_to_anchor=(1.0, 1.0))
    ax[i,0].axvline(x=t1, ls='--', color='grey')
    ax[i,0].axvline(x=t2, ls='--', color='grey')
    #ax[i,0].set_xlim(t1, t2)
    #ax[i,0].xaxis.set_major_locator(mdates.MonthLocator(interval=24))
    #ax[i,0].xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    #ax[i,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if i>=len(output_targets):
        ax[i,1].axis('off')

for j,col in enumerate(output_targets):
    ax[j,1].plot(df[col], label=col)
    ax[j,1].legend(loc='upper right',bbox_to_anchor=(1.0, 1.0))
    ax[j,1].axvline(x=t1, ls='--', color='grey')
    ax[j,1].axvline(x=t2, ls='--', color='grey')
    #ax[j,1].set_xlim(t1, t2)
    #ax[j,1].xaxis.set_major_locator(mdates.MonthLocator(interval=24))
    #ax[j,1].xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    #ax[j,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if j>=len(input_features):
        ax[j,0].axis('off')

ax[0,0].set_title('Input features')

ax[0,1].set_title('Output targets')

plt.show()


def f(X,mid):
    Y = mid + (X - X.mean()) / (X.max() - X.min())
    return Y

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
plt.subplots_adjust(hspace=0.25)

ax.axvspan(ymin=0,ymax=1, xmin=df.index[id_train_beg],xmax=df.index[id_train_end],label='train',color='green',alpha=0.5)
ax.axvspan(ymin=0,ymax=1, xmin=df.index[id_test_beg],xmax=df.index[id_test_end],label='test',color='orange',alpha=0.5)
ax.axvspan(ymin=0,ymax=1, xmin=df.index[id_val_beg],xmax=df.index[id_val_end],label='validation',color='blue',alpha=0.5)



mids = 1*(np.linspace(0,1,4)[1:]+np.linspace(0,1,4)[:-1])/2

ax.axhline(mids[0],color='black',linestyle='--')
ax.axhline(mids[1],color='black',linestyle='--')
ax.axhline(mids[2],color='black',linestyle='--')
ax.plot(df.index.values,1*f(df['ex'],mids[-1]),label='ex')
ax.plot(df.index.values,1*f(df['ey'],mids[-2]),label='ey')
ax.plot(df.index.values,1*f(df['ez'],mids[-3]),label='ez')

ax.set_yticks(mids,[f'{df[e].mean():.2f}' for e in ['ex','ey','ez']])
ax.set_ylim(0,1)
fig.legend()

fig.subplots_adjust(top=0.82)
fig.subplots_adjust(right=1.15)
fig.suptitle("Data partitioning", fontsize=20)

plt.show()




