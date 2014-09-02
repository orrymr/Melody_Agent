package tablebasedagent;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.logging.Level;
import java.util.logging.Logger;

/*
 * To change this template, choose Tools | Templates and open the template in
 * the editor.
 */
/**
 *
 * @author orrymr
 */
public class TxtToTrajectory {

    int[] start_state = {-1, -1, -1, -1};
    int BARS = 8, BEATS = 8;

    public static void main(String[] args) {
        for(int i = 1; i <= 20; i++ ){
            TxtToTrajectory txt = new TxtToTrajectory(new File("DMaj/mel" + i +".txt"));
        }
    }

    public TxtToTrajectory(File txtFile) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(txtFile));

            int s [] = start_state;
            
            String line = null;
            StringBuilder greedyP = new StringBuilder();
            
            while ((line = br.readLine()) != null) {
                String[] components = line.split(",");
                int a = Integer.parseInt(components[2]);
                
                greedyP.append("([" + createKey(s) + "], ");
                
                //To make D Sharp..
                a++;
                if(a == 26)
                    a = 2;
                
                greedyP.append(a + ")#");
                s = performAction(a, s);
            }

            greedyP.append("([" + createKey(s) + "], -1)");
            
            System.out.println(greedyP.toString());

        } catch (Exception ex) {
            Logger.getLogger(TxtToTrajectory.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static String createKey(int[] s) {
        String key = "";
        for (int i = 0; i < s.length; i++) {
            int j = s[i];
            key += j + " ";
        }
        key = key.trim();
        return key;
    }

    public static int[] performAction(int a, int[] s) {
        int[] new_s = s.clone();
        for (int i = s.length - 1; i > 0; i--) {
            new_s[i] = new_s[i - 1];
        }
        new_s[0] = a;

        return new_s;
    }
}