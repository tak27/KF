classdef KF
    %KF カルマンフィルタクラス
    properties
        model % A,B,G,H,R,history
        x % 状態値
        P % 誤差共分散
        K % カルマンゲイン
        x_hist % 状態値x_{k|k}の履歴
        P_hist % 状態値の誤差共分散P_{k|k}の履歴
        xp_hist % 予測値x_{k|k-1}の履歴
        Pp_hist % 予測値の誤差共分散P_{k|k-1}の履歴
    end
    methods
        function kf=KF(x0,P0,varargin)
            %KF カルマンフィルタを初期化する。
            %    kf=KF(x0,P0): 初期の状態値をx0、誤差共分散をP0としてカルマンフィルタkfを初期化する。
            %    kf=KF(x0,P0,model): 初期の状態値をx0、誤差共分散をP0、モデルをmodelとしてカルマンフィルタkfを初期化する。
            kf.x=x0;
            kf.P=P0;
            if 1<=length(varargin)
                kf.model=varargin{1};
                if isfield(kf.model,'history') && kf.model.history
                    kf.x_hist=x0;
                    kf.P_hist=diag(P0);
                end
            end
        end
        function pred=predict(kf,varargin)
            %PREDICT 次の時刻の状態値を予測する。
            %    pred=kf.predict(): 空の操作量が与えた次の時刻の状態値pred.xを予測し、誤差共分散行列pred.Pを与える。
            %    pred=kf.predict(u): 操作量uを与えた次の時刻の状態値pred.xを予測し、誤差共分散行列pred.Pを与える。
            %    pred=kf.predict(u,model): 操作量uとモデルmodelを与えた次の時刻の状態値pred.xを予測し、誤差共分散行列pred.Pを与える。
            %
            %モデルmodelは、オーバーライドしたい行列だけ含めば良いものとする。
            if 1<=length(varargin)
                u=varargin{1};
            else
                u=kf.null_u();
            end
            if 2<=length(varargin)
                kf.model=overwritestruct(kf.model,varargin{2});
            end
            pred=struct(...
                'x',kf.model.A*kf.x+kf.model.B*u,...
                'P',kf.model.A*kf.P*kf.model.A'+kf.model.G*kf.model.Q*kf.model.G');
        end
        function [kf,x,P,K,pred]=update(kf,z,varargin)
            %UPDATE 状態値を更新する。
            %    [kf,x,P,K,pred]=kf.update(z): 観測値zを与えた現在の状態値を更新し、状態値x、誤差共分散行列P、カルマンゲインK、事前予測値predを返す。
            %    [kf,x,P,K,pred]=kf.update(z,pred): 事前予測値predと観測値zを与えた現在の状態値を更新し、状態値x、誤差共分散行列P、カルマンゲインK、事前予測値predを返す。
            %    [kf,x,P,K,pred]=kf.update(z,pred,model): モデルmodelと事前予測値pred、観測値zを与えた現在の状態値を更新し、状態値x、誤差共分散行列P、カルマンゲインK、事前予測値predを返す。
            %
            %モデルmodelは、オーバーライドしたい行列だけ含めば良いものとする。
            if 1<=length(varargin)
                pred=varargin{1};
            else
                pred=kf.predict();
            end
            if 2<=length(varargin)
                kf.model=overwritestruct(kf.model,varargin{2});
            end
            K=pred.P*kf.model.H'/(kf.model.R+kf.model.H*pred.P*kf.model.H');
            KH=K*kf.model.H;
            I=eye(size(KH));
            x=pred.x+K*(z-kf.model.H*pred.x);
            P=(I-KH)*pred.P;
            
            kf.x=x;
            kf.P=P;
            kf.K=K;
            if isfield(kf.model,'history') && kf.model.history
                kf.x_hist=[kf.x_hist x];
                kf.P_hist=[kf.P_hist diag(P)];
                kf.xp_hist=[kf.xp_hist pred.x];
                kf.Pp_hist=[kf.xp_hist diag(pred.P)];
            end
        end
        function u=null_u(kf)
            %NULL_U 空の入力を作成する。
            u=zeros(size(kf.model.B,2),1);
        end
        function [dist,conf]=eval_dist(kf,z,varargin)
            %EVAL_DIST 予測値と観測値の距離を評価する。
            %    [dist,conf]=kf.eval_dist(z): 予測値と観測値zの距離distと確信度（距離を標準偏差で割った値のノルム）confを得る。
            %    [dist,conf]=kf.eval_dist(z,pred): 予測値predを用いる。
            %    [dist,conf]=kf.eval_dist(z,pred,model): 予測値predとモデルmodelを用いる。
            %
            %観測値zは1個以上の縦ベクトルとする。
            %距離distと確信度confはそれぞれ、観測値zに対応する1個以上の縦ベクトルおよびスカラ値の配列とする。
            if 1<=length(varargin)
                pred=varargin{1};
            else
                pred=kf.predict();
            end
            if 2<=length(varargin)
                kf.model=overwrite(kf.model,varargin{2});
            end
            dist=z-kf.model.H*pred.x;
            devi=kf.model.H*sqrt(diag(pred.P));
            conf=vecnorm(dist./devi,2,1);
        end
    end
end
