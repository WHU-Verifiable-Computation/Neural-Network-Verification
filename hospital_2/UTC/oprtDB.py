import pandas as pd


class oprtDB:
    def deleteCSV(self,bid,path):
        bid = int(bid)
        df = pd.read_csv(path,float_precision='round_trip').drop(bid)
        df.to_csv(path,header=["V","T"],index=False)

    def changeCSV(self,bid,path,VSig,TFile):
        bid = int(bid)
        df = pd.read_csv(path,float_precision='round_trip')
        df.loc[bid,'V'] = VSig
        df.loc[bid, 'T'] = TFile
        df.to_csv(path, header=["V", "T"], index=False)

    def insertCSV(self, bid, path,VSig,TFile):
        df = pd.read_csv(path, float_precision='round_trip')
        bid = int(bid)

        df_pre = pd.read_csv(path, float_precision='round_trip').head(bid)
        df_pre.loc[len(df_pre)] = [VSig,TFile]
        df_post = pd.read_csv(path, float_precision='round_trip').tail(len(df)-bid)

        df_res = df_pre.append(df_post)

        df_res.to_csv(path,header=["V","T"],index=False)

# if __name__ == '__main__':
#     o = oprtDB()
#     o.insertCSV(5,"/media/yorio/61B6-9D42/Data/TPA/11.CSV","9999","9999")