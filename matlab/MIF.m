classdef MIF
    %MIF �������͂̃t�B���^
    properties
        model % A,B,G,H,R,history
        threshold % threshold for the distance between a observation value and a state value predicted by a filter to determing if the observation value is one which should be bound to the filter.
        filters % �t�B���^�̃Z���z��
    end
    methods
        function mif=MIF(model,threshold)
            mif.model=model;
            mif.threshold=threshold;
        end
        function preds=predict(mif,varargin)
            %PREDICT �e�t�B���^�Ŏ��̎����X�e�b�v�ɂ������Ԓl��\������B
            %    preds=mif.predict(): ���̎����X�e�b�v�ɂ������Ԓlpreds��\������B
            %    preds=mif.predict(u): �����u��p����B
            %    preds=mif.predict(u,model): �����u�ƃ��f��model��p����B
            %
            %�\���lpreds�A�����u�͋��Ƀt�B���^���̃Z���z��Ƃ��Amif.filters�Ɠ���̕��тƂ���B
            %���f��model�́A�I�[�o�[���C�h�������s�񂾂��܂߂Ηǂ����̂Ƃ���B
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
            %UPDATE �ϑ��l����Ɋe�t�B���^�̌��ݎ����̏�Ԓl�𐄒肷��B
            %    mif=mif.update(z): �ϑ��lz����Ɋe�t�B���^�̌��ݎ����̏�Ԓl�𐄒肷��B
            %    mif=mif.update(z,preds): �\���lpreds��p����B
            %    mif=mif.update(z,preds,model): �\���lpreds�ƃ��f��model��p����B
            %    [mif,x,P,K]=mif.update(__): ����lx�A�덷�����UP�A�J���}���Q�C��K��Ԃ��B
            %
            %�ϑ��lz�͑S�Z���T�̊ϑ��l�̏c�x�N�g������Ȃ鏇�s���̔z��Ƃ���B
            %�\���lpreds�̓t�B���^���̃Z���z��Ƃ��Amif.filters�Ɠ���̕��тƂ���B
            %����lx�A�덷�����UP�A�J���}���Q�C��K�̓t�B���^���̃Z���z��Ƃ��Amif.filters�Ɠ���̕��тƂ���B
            %���f��model�́A�I�[�o�[���C�h�������s�񂾂��܂߂Ηǂ����̂Ƃ���B
            nz=size(z,2);
            nobj=length(mif.filters);
            %���f�����w�肳��Ă���ꍇ��MIF���̃��f�����X�V����B
            if 2<=length(varargin)
                mif.model=overwritestruct(mif.model,varargin{2});
            end
            %�\���l���w�肳��Ă��Ȃ���Ζ�����ʂ̗\�������s����B
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

