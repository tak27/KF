function s1 = overwritestruct(s1,s2,varargin)
%OVERWRITESTRUCT 構造体のフィールド値を上書きする。
%   s1=overwritestruct(s1,s2): 構造体s2のフィールド値で構造体s1のフィール値を上書きする。
%   s1=overwritestruct(s1,s2,true): 両方のフィール値が構造体の場合、同様にフィールド値を上書きする。
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

