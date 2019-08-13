function s1 = overwritestruct(s1,s2,varargin)
%OVERWRITESTRUCT �\���̂̃t�B�[���h�l���㏑������B
%   s1=overwritestruct(s1,s2): �\����s2�̃t�B�[���h�l�ō\����s1�̃t�B�[���l���㏑������B
%   s1=overwritestruct(s1,s2,true): �����̃t�B�[���l���\���̂̏ꍇ�A���l�Ƀt�B�[���h�l���㏑������B
deep=(1<=length(varargin) && varargin{1});
if isempty(s1) || ~isstruct(s2)
    s1=s2;
else
    for f=fieldnames(s2)'
        name=f{1};
        if deep && isfield(s1,name)
            s1.(name)=overwritestruct(s1.(name),s2.(name));
        else
            s1.(name)=s2.(name);
        end
    end
end
end

