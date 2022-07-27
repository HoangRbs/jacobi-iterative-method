# type "Main" inside the "Command Window" tab, run with octave or mathlab

function Main()
   N = 25;
   M = 25;
   tolerance = 0.00001;
   c0 = 0.25;
   cL = 0.25;
   
   conMatx = zeros(N, M);
   maxConDiff = 0; prevCon = 0;
   west = 0; 
   east = 0; 
   south = 0; 
   north = 0;
   k = 0;
   while 1
      
      maxConDiff = 0;
      prevCon = 0;
      
      for i = (1: N)
         for j = (1: M)
            prevCon = conMatx(i, j);
            
            if i == N
               conMatx(i, j) = 1;
            elseif i == 1
               conMatx(i, j) = 0;
            else
               if j == 1
                  west = c0;   
               else
                  west = conMatx(i, j-1);
               end
               
               if j == M
                  east = cL;
               else
                  east = conMatx(i, j+1);
               end
               
               north = conMatx(i - 1, j);
               south = conMatx(i + 1, j);
               
               conMatx(i, j) = 0.25 * (west + east + north + south);
            end
            
            if abs(conMatx(i, j) - prevCon) >= tolerance 
               maxConDiff = abs(conMatx(i, j) - prevCon);
            endif
         endfor
      endfor
      
      if maxConDiff < tolerance
         break;
      endif
      
   endwhile
   
   imagesc(conMatx);
endfunction