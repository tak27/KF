classdef KF
    %KF �J���}���t�B���^�N���X
    properties
        model % A,B,G,H,R,history
        x % ��Ԓl
        P % �덷�����U
        K % �J���}���Q�C��
        x_hist % ��Ԓlx_{k|k}�̗���
        P_hist % ��Ԓl�̌덷�����UP_{k|k}�̗���
        xp_hist % �\���lx_{k|k-1}�̗���
        Pp_hist % �\���l�̌덷�����UP_{k|k-1}�̗���
    end
    methods
        function kf=KF(x0,P0,varargin)
            %KF �J���}���t�B���^������������B
            %    kf=KF(x0,P0): �����̏�Ԓl��x0�A�덷�����U��P0�Ƃ��ăJ���}���t�B���^kf������������B
            %    kf=KF(x0,P0,model): �����̏�Ԓl��x0�A�덷�����U��P0�A���f����model�Ƃ��ăJ���}���t�B���^kf������������B
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
            %PREDICT ���̎����̏�Ԓl��\������B
            %    pred=kf.predict(): ��̑���ʂ��^�������̎����̏�Ԓlpred.x��\�����A�덷�����U�s��pred.P��^����B
            %    pred=kf.predict(u): �����u��^�������̎����̏�Ԓlpred.x��\�����A�덷�����U�s��pred.P��^����B
            %    pred=kf.predict(u,model): �����u�ƃ��f��model��^�������̎����̏�Ԓlpred.x��\�����A�덷�����U�s��pred.P��^����B
            %
            %���f��model�́A�I�[�o�[���C�h�������s�񂾂��܂߂Ηǂ����̂Ƃ���B
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
            %UPDATE ��Ԓl���X�V����B
            %    [kf,x,P,K,pred]=kf.update(z): �ϑ��lz��^�������݂̏�Ԓl���X�V���A��Ԓlx�A�덷�����U�s��P�A�J���}���Q�C��K�A���O�\���lpred��Ԃ��B
            %    [kf,x,P,K,pred]=kf.update(z,pred): ���O�\���lpred�Ɗϑ��lz��^�������݂̏�Ԓl���X�V���A��Ԓlx�A�덷�����U�s��P�A�J���}���Q�C��K�A���O�\���lpred��Ԃ��B
            %    [kf,x,P,K,pred]=kf.update(z,pred,model): ���f��model�Ǝ��O�\���lpred�A�ϑ��lz��^�������݂̏�Ԓl���X�V���A��Ԓlx�A�덷�����U�s��P�A�J���}���Q�C��K�A���O�\���lpred��Ԃ��B
            %
            %���f��model�́A�I�[�o�[���C�h�������s�񂾂��܂߂Ηǂ����̂Ƃ���B
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
            %NULL_U ��̓��͂��쐬����B
            u=zeros(size(kf.model.B,2),1);
        end
        function [dist,conf]=eval_dist(kf,z,varargin)
            %EVAL_DIST �\���l�Ɗϑ��l�̋�����]������B
            %    [dist,conf]=kf.eval_dist(z): �\���l�Ɗϑ��lz�̋���dist�Ɗm�M�x�i������W���΍��Ŋ������l�̃m�����jconf�𓾂�B
            %    [dist,conf]=kf.eval_dist(z,pred): �\���lpred��p����B
            %    [dist,conf]=kf.eval_dist(z,pred,model): �\���lpred�ƃ��f��model��p����B
            %
            %�ϑ��lz��1�ȏ�̏c�x�N�g���Ƃ���B
            %����dist�Ɗm�M�xconf�͂��ꂼ��A�ϑ��lz�ɑΉ�����1�ȏ�̏c�x�N�g������уX�J���l�̔z��Ƃ���B
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
