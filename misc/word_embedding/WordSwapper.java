/* This code is for swapping a list of words at a given possibility in a corpus.
*
*  License: BSD
*  Author: Sen Jia
*/


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedInputStream;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.lang.Math;
import java.nio.file.Files;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.stream.Stream;
import java.nio.file.Paths;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedWriter;
import java.io.File;
import java.io.OutputStream;
import java.io.FileWriter;

public class WordSwapper{
    public static void main(String[] args){
        try{
       
		String swap_file = "XXX.txt";  // a file contains word pairs you want to swap and their swapping possibility, [word1 word2 possibility] TAB separated.
		BufferedReader br = new BufferedReader(new FileReader(swap_file));
		String line;
 
                ArrayList<String> word_list_1 = new ArrayList<String>();
                ArrayList<String> word_list_2 = new ArrayList<String>();
                ArrayList<Float> probability = new ArrayList<Float>();

		while ((line = br.readLine()) != null) {
                    String[] strs = line.split("\t");
                    word_list_1.add(strs[0]);
                    word_list_2.add(strs[1]);
                    probability.add(Float.valueOf(strs[2]));
		}
                
        
                String input_file = "YYY.txt"; // the corpus file you want to modify.
                FileReader inputStream = new FileReader(input_file);


                String output_file = "ZZZ.txt";  // the output file for your modified corpus.
                BufferedWriter out = new BufferedWriter(new FileWriter(output_file));
                
                int c;
                StringBuilder word_builder = new StringBuilder();

		while ((c = inputStream.read()) != -1) {
		    char ch = (char) c;
		    if (ch != ' '){
		        word_builder.append(ch); 
		    }else{
		        if (word_builder.length()>0){
		             String word = word_builder.toString();
			     if (word_list_1.contains(word)){
			         int index = word_list_1.indexOf(word);
			         double r = Math.random();
			         if (r < probability.get(index)){
			   	     word = word_list_2.get(index);  
				 }
			     }else if (word_list_2.contains(word)){
			         int index = word_list_2.indexOf(word);
				 double r = Math.random();
				 if (r < probability.get(index)){
				     word = word_list_1.get(index);  
				 }
			     }
		    out.write(word+" ");
		    word_builder.setLength(0);
			}
	            }
			
		}

            }catch(Exception e){
                e.printStackTrace();
            }
       }

}
