function [ mu ] = func_mean( x, data )
%FUNC_MEAN Summary of this function goes here
%   Detailed explanation goes here

mu = zeros(length(x)*5, 1);
for i = 1:5
    switch i
        case 1
            for j = 1:length(x)
                numHour = hour(x(j)) + 1;
                numWeek = week(x(j));
                if(weekday(x(j)) == 1)
                    numWeekday = 7;
                else
                    numWeekday = weekday(x(j)) - 1;
                end
                mu((i-1)*length(x)+j) = mean(data(numHour, numWeekday, numWeek:52:end, i));
            end
        otherwise
            for j = 1:length(x)
                numHour = hour(x(j)) + 1;
                numWeek = week(x(j));
                mu((i-1)*length(x)+j) = mean(data(numHour, :, numWeek, i));
            end
    end
end

end

