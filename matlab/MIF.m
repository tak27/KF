classdef MIF
    %MIF 複数入力のフィルタ
    properties
        model % A,B,G,H,R,history
        threshold % threshold for the distance between a observation value and a state value predicted by a filter to determing if the observation value is one which should be bound to the filter.
        filters % フィルタのセル配列
    end
    methods
        function mif=MIF(model,threshold)
            mif.model=model;
            mif.threshold=threshold;
        end
        function preds=predict(mif,varargin)
            %PREDICT 各フィルタで次の時刻ステップにおける状態値を予測する。
            %    preds=mif.predict(): 次の時刻ステップにおける状態値predsを予測する。
            %    preds=mif.predict(u): 操作量uを用いる。
            %    preds=mif.predict(u,model): 操作量uとモデルmodelを用いる。
            %
            %予測値preds、操作量uは共にフィルタ毎のセル配列とし、mif.filtersと同一の並びとする。
            %モデルmodelは、オーバーライドしたい行列だけ含めば良いものとする。
            nobj=length(mif.filters);
            if 2<=length(varargin)
                mif.model=overwritestruct(mif.model,varargin{2});
            end
            u=[];
            if 1<=length(varargin)
                u=varargin{1};
            end
            if isempty(u)
                u=cell(1,nobj);
                for i=1:nobj
                    u{i}=mif.filters{i}.null_u();
                end
            end
            preds=cell(1,nobj);
            for i=1:nobj
                preds{i}=mif.filters{i}.predict(u{i},mif.model);
            end
        end
        function [mif,varargout]=update(mif,z,varargin)
            %UPDATE 観測値を基に各フィルタの現在時刻の状態値を推定する。
            %    mif=mif.update(z): 観測値zを基に各フィルタの現在時刻の状態値を推定する。
            %    mif=mif.update(z,preds): 予測値predsを用いる。
            %    mif=mif.update(z,preds,model): 予測値predsとモデルmodelを用いる。
            %    [mif,x,P,K]=mif.update(__): 推定値x、誤差共分散P、カルマンゲインKを返す。
            %
            %観測値zは全センサの観測値の縦ベクトルからなる順不同の配列とする。
            %予測値predsはフィルタ毎のセル配列とし、mif.filtersと同一の並びとする。
            %推定値x、誤差共分散P、カルマンゲインKはフィルタ毎のセル配列とし、mif.filtersと同一の並びとする。
            %モデルmodelは、オーバーライドしたい行列だけ含めば良いものとする。
            nz=size(z,2);
            nobj=length(mif.filters);
            %モデルが指定されている場合はMIF内のモデルを更新する。
            if 2<=length(varargin)
                mif.model=overwritestruct(mif.model,varargin{2});
            end
            %予測値が指定されていなければ無操作量の予測を実行する。
            if 1<=length(varargin)
                preds=varargin{1};
            else
                preds=mif.predict();
            end
            ztof=zeros(1,nz);
            ftoz=zeros(1,nobj);
            for i=1:nobj
                [~,conf]=mif.filters{i}.eval_dist(z,preds{i});
                [~,j]=min(conf);
                if conf(j)<=mif.threshold
                    ztof(j)=i;
                    ftoz(i)=j;
                end
            end
            ext_nobj=nobj;
            for j=1:nz
                if ztof(j)==0
                    ext_nobj=ext_nobj+1;
                    ztof(j)=ext_nobj;
                    ftoz(ztof(j))=j;
                end
            end
            if nobj<ext_nobj
                mif.filters{ext_nobj}=[];
            end
            for i=1:ext_nobj
                if isempty(mif.filters{i})
                    mif.filters{i}=KF(z(:,ftoz(i)),mif.model.R,mif.model);
                elseif 1<=ftoz(i)
                    mif.filters{i}=mif.filters{i}.update(z(:,ftoz(i)),preds{i},mif.model);
                end
            end
            if ~isfield(mif.model,'history') || ~mif.model.history
                mif.filters(ftoz==0)=[];
            end
            if 2<=nargout
                x=cell(1,length(mif.filters));
                for i=1:length(mif.filters)
                    x{i}=mif.filters{i}.x;
                end
                varargout{1}=x;
            end
            if 3<=nargout
                P=cell(1,length(mif.filters));
                for i=1:length(mif.filters)
                    P{i}=mif.filters{i}.P;
                end
                varargout{2}=P;
            end
            if 4<=nargout
                K=cell(1,length(mif.filters));
                for i=1:length(mif.filters)
                    K{i}=mif.filters{i}.K;
                end
                varargout{3}=K;
            end
        end
    end
    
end

