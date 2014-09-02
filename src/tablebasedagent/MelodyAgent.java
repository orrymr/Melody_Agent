/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package tablebasedagent;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.*;

/**
 *
 * @author orrymr
 */
public class MelodyAgent {

    static final int NUM_ACTIONS = 25;
    static final int BEATS = 8;
    static final int BARS = 8;
    static final double EPSILON = 0.2; //For epsilon-greedy
    static final double LAMBDA = 0.0;
    final static double ALPHA = 0.1; //Learning rate
    final static double GAMMA = 0.9; //discount factor for sarsa
    final static double GAMMA_2 = 0.9; //discount factor for calculating feature expectations
    static final int[] start_state = {-1, -1, -1, -1};
    static final int STATE_SIZE = 4;
    static Map<String, Double> q[];
    static HashMap<String, Double>[] e = new HashMap[NUM_ACTIONS];
    static String goalState = "1 2 3 4";
    static File trajFile = new File("All.txt");
    static boolean GREEDY = false;
    static final double TEMP = 0.25;

    public static void createWeights(Map<String, Double> w) {
        w.put(goalState, 100.0);
    }

    public static void main(String[] args) {
        //exp1();
        exp2();
    }

    public static void exp1() {
        //Does RL - uses Sarsa/SarsaLambda without RBFs - it's just table based
        GREEDY = true;
        Map<String, Double> w = new HashMap();
        createWeights(w);
        resetQ();
        qLearningLambda(w);
        //sarsaLambda(w);
        playGame();
    }

    public static void exp2() {
        //Does AL Via IRL - table based!
        GREEDY = false;
        double EPSILON = 0.01;

        LinkedList<Map<String, Integer>> pi = new LinkedList();

        Map<String, Double> Mu_E = new HashMap();
        LinkedList<Map<String, Double>> Mu = new LinkedList();
        LinkedList<Map<String, Double>> Mu_bar = new LinkedList();
        LinkedList<Map<String, Double>> w = new LinkedList();


        //The pre-processing bit. This computes expert feature expectations. Part of this method is to actually create
        //the expert trajectories. In it, we actually do sarsa; play the game to get the trajectories
        computeMu_E(Mu_E);

        for (Map.Entry<String, Double> entry : Mu_E.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }

        //reset the q-tables
        resetQ();

        //Step 1
        //first need to compute trajectories for the random policy
        String[] traj = retrieveTraj();
        //Only then can the feature expections be calcuated, and added to the LinkedList, as per the next line:
        Mu.add(computeMu(traj));

        int i = 1;
        while (true) {
            //step 2 - Projection Method
            Mu_bar.add(computeMuBar(Mu_E, Mu, Mu_bar, i));
            w.add(subtractVectors(Mu_E, Mu_bar.get(i - 1))); //set w(i)
            normalizeWeights(w.getLast());
            double t = dot(w.get(i - 1), w.get(i - 1)); //set t(i)
            System.out.println("");
            System.out.println("t: " + t);
            
            
            /*for (Map.Entry<String, Double> entry : w.get(i - 1).entrySet()) {
                if (entry.getValue() > 0) {
                    System.out.println(entry.getKey() + "-> " + entry.getValue());
                }
            }
            playGame();*/
            //step 3 - termination condition
            if (t <= EPSILON) //Note: this is the locally define EPSILON... 
            {
                break;
            }

            //step 4 - solve the MDP
            //resetQ();

            qLearningLambda(w.getLast());
            traj = retrieveTraj();

            //step 5
            Mu.add(computeMu(traj));

            //step 6
            i++;
        }
        System.out.println("");
        System.out.println("i: " + i);
        System.out.println(">>>>>>>>>>>>>>>>>>>WEIGHTS<<<<<<<<<<<<<<<<<<<<<");
        for (Map.Entry<String, Double> entry : w.get(i - 2).entrySet()) {
            if (entry.getValue() > 0) {
                System.out.println(entry.getKey() + "-> " + entry.getValue());
            }
        }

        System.out.println("Final EVER Greedy: ");
        String greedyP = playGame2();
        System.out.println(greedyP);
        convert(greedyP);

        System.out.println("Final EVER Softmax: ");
        String softP = playSoftGame();
        System.out.println(softP);
        convert(softP);
    }

    public static void normalizeWeights(Map<String, Double> w) {
        double total = 0;

        for (Map.Entry<String, Double> entry : w.entrySet()) {
            if (entry.getValue() > 0) {
                total += entry.getValue();
            }
        }

        for (Map.Entry<String, Double> entry : w.entrySet()) {
            if (entry.getValue() > 0) {
                double val = entry.getValue() / total;
                entry.setValue(val);
            }
        }
    }

    public static Map<String, Double> computeMuBar(Map<String, Double> Mu_E, LinkedList<Map<String, Double>> Mu, LinkedList<Map<String, Double>> Mu_bar, int i) {
        Map<String, Double> mu_bar = new HashMap();

        if (i == 1) {
            return Mu.get(0);
        } else {
            mu_bar = addVectors(Mu_bar.get(i - 2), secondTerm(Mu.get(i - 1), Mu_bar.get(i - 2), Mu_E));
        }

        return mu_bar;
    }

    public static Map<String, Double> addVectors(Map<String, Double> term1, Map<String, Double> term2) {
        Map<String, Double> result = new HashMap();
        makeHashMapsCorrespond(term1, term2);

        for (Map.Entry<String, Double> entry : term1.entrySet()) {
            result.put(entry.getKey(), term1.get(entry.getKey()) + term2.get(entry.getKey()));
        }

        return result;
    }

    public static Map<String, Double> secondTerm(Map<String, Double> mu_i_minus_one, Map<String, Double> mu_bar_i_minus_two, Map<String, Double> Mu_E) {
        Map<String, Double> secondterm = new HashMap();
        double numerator, denomenator, coefficient;

        Map<String, Double> term1 = subtractVectors(mu_i_minus_one, mu_bar_i_minus_two);
        Map<String, Double> term2 = subtractVectors(Mu_E, mu_bar_i_minus_two);

        numerator = dot(term1, term2);
        denomenator = dot(term1, term1);

        coefficient = numerator / denomenator;

        for (Map.Entry<String, Double> entry : term1.entrySet()) {
            double value = coefficient * entry.getValue();
            secondterm.put(entry.getKey(), value);
        }

        return secondterm;
    }

    public static int distance(String s1, String s2) {
        s1 = s1.replaceAll("\\s", "");
        s2 = s2.replaceAll("\\s", "");
        int dist = 0; //a distance of 0 means that the 2 states are the same.
        String temp_s1 = s1;
        String temp_s2 = s2;

        for (int i = 0; i < s1.length(); i++) {
            if (temp_s1.equals(temp_s2)) {
                return dist;
            }
            dist++;
            temp_s1 = s1.substring(dist);
            temp_s2 = s2.substring(0, s2.length() - dist);
        }

        return -1;// a distance of -1 is the furthest away 2 states can be.
    }

    public static double dot(Map<String, Double> term1, Map<String, Double> term2) {
        Double result = Double.NaN;

        makeHashMapsCorrespond(term1, term2);

        for (Map.Entry<String, Double> entry : term1.entrySet()) {
            if (result.isNaN()) {
                result = term1.get(entry.getKey()) * term2.get(entry.getKey());
            } else {
                result += term1.get(entry.getKey()) * term2.get(entry.getKey());
            }
        }

        return result;
    }

    public static Map<String, Double> subtractVectors(Map<String, Double> vector1, Map<String, Double> vector2) {
        Map<String, Double> result = new HashMap();

        makeHashMapsCorrespond(vector1, vector2);
        for (Map.Entry<String, Double> entry : vector1.entrySet()) {
            double difference = vector1.get(entry.getKey()) - vector2.get(entry.getKey());
            result.put(entry.getKey(), difference);
        }

        return result;
    }

    public static void makeHashMapsCorrespond(Map<String, Double> map1, Map<String, Double> map2) {
        //muBar = map1
        //mu = map2

        //if map2 doesn't contain any value for the corresponding map1, then we put a 0, so that it has something to subtract
        for (Map.Entry<String, Double> entry : map1.entrySet()) {
            if (!map2.containsKey(entry.getKey())) {//if the corresponding mu doesn't contain an entry for the corresponding muBar
                map2.put(entry.getKey(), 0.0);
            }
        }

        //now do it the other way round
        //that is, map1 must contain the corresponding value for map2
        for (Map.Entry<String, Double> entry : map2.entrySet()) {
            if (!map1.containsKey(entry.getKey())) {
                map1.put(entry.getKey(), 0.0);
                //System.out.println("if map1 doesn't contain the key/value combo, add it in");
            }
        }
    }

    public static void computeMu_E(Map<String, Double> Mu_E) {
        int numT = getNumT();
        System.out.println("Number of Expert trajectories: " + numT);
        LinkedList<StateAction> traj[] = getTraj(numT); //the array size should = number of traj's    

        for (LinkedList<StateAction> tr : traj) {//sum over all m
            int t = 0;
            for (StateAction sa : tr) {//sum over all t
                int action = sa.getAction();
                int[] state = sa.getState();

                String stateKey = createKey(state);

                if (stateKey.equals(createKey(start_state))) {
                    continue;
                }

                double val = Mu_E.containsKey(stateKey) ? Mu_E.get(stateKey) : 0;

                val += Math.pow(GAMMA_2, t);

                Mu_E.put(stateKey, val);

                t++;
            }
        }

        for (Map.Entry<String, Double> entry : Mu_E.entrySet()) {
            double val = entry.getValue();
            val = val / (double) numT;
            entry.setValue(val);
        }
    }

    public static int getNumT() {
        int num = -1;
        try {
            FileReader fr = new FileReader(trajFile);
            BufferedReader br = new BufferedReader(fr);

            num = Integer.parseInt(br.readLine());
        } catch (Exception e) {
            e.printStackTrace();
        }

        return num;
    }

    public static LinkedList<StateAction>[] getTraj(int numT) {
        LinkedList<StateAction>[] trajs = new LinkedList[numT];

        for (int i = 0; i < trajs.length; i++) {
            trajs[i] = new <StateAction> LinkedList();
        }

        try {
            FileReader fr = new FileReader(trajFile);
            BufferedReader br = new BufferedReader(fr);
            br.readLine();
            for (int i = 0; i < numT; i++) {
                String traj = br.readLine();
                String[] statesAndAction = traj.split("#");

                for (int step = 0; step < statesAndAction.length; step++) {
                    String string = statesAndAction[step];
                    string = string.substring(1, string.length() - 1);
                    String[] sa = string.split(",");
                    sa[1] = sa[1].trim();
                    sa[0] = sa[0].substring(1, sa[0].length() - 1);

                    String[] stateString = sa[0].split(" ");
                    int[] state = new int[STATE_SIZE];

                    for (int k = 0; k < STATE_SIZE; k++) {
                        state[k] = Integer.parseInt(stateString[k]);
                    }

                    int action = Integer.parseInt(sa[1]);

                    StateAction sa_obj = new StateAction(state, action);
                    trajs[i].add(sa_obj);
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return trajs;
    }

    public static void computeMu_E(Map<String, Double> Mu_E, int m) {
        //m refers to the number of trajectories to create
        //this method actually creates the trajectories first, before calculating Mu_E
        Map<String, Double> mus[] = new HashMap[m];
        String[][] trajectories = new String[m][BEATS * BARS];
        Map<String, Double> w = new HashMap();
        createWeights(w);

        //First need to create expert trajectories
        for (int i = 0; i < m; i++) {
            //Intialize will reset the q-values to 0
            resetQ();
            qLearningLambda(w);

            trajectories[i] = retrieveTraj();
        }

        //print out the expert t's
        for (int i = 0; i < m; i++) {
            System.out.println("Trajectory num: " + i);
            for (int j = 0; j < BEATS * BARS; j++) {
                System.out.print(trajectories[i][j] + ", ");
            }
            System.out.println("");
        }

        //Create the Mu for each trajectory and store it in an array
        for (int i = 0; i < m; i++) {
            mus[i] = computeMu(trajectories[i]);
        }

        //Now average the mus to get to Mu_E
        for (int i = 0; i < m; i++) {
            for (Map.Entry<String, Double> entry : mus[i].entrySet()) {
                double value = Mu_E.containsKey(entry.getKey()) ? Mu_E.get(entry.getKey()) : 0;
                value += entry.getValue();
                Mu_E.put(entry.getKey(), value);
            }
        }

        for (Map.Entry<String, Double> entry : Mu_E.entrySet()) {
            double value = entry.getValue() / (double) m;
            Mu_E.put(entry.getKey(), value);
        }
    }

    public static Map computeMu(String[] traj) {
        Map<String, Double> mu = new HashMap();
        int t = 0;

        for (int i = 0; i < BARS; i++) {
            for (int j = 0; j < BEATS; j++) {

                if (mu.containsKey(traj[t])) {
                    double val = mu.get(traj[t]);
                    val += 1 * Math.pow(GAMMA_2, t);
                    mu.put(traj[t], val);
                } else {
                    mu.put(traj[t], Math.pow(GAMMA_2, t));
                }

                t++;
            }
        }
        return mu;
    }

    public static String[] retrieveTraj() {
        int[] s = start_state;
        String[] traj = new String[BEATS * BARS];
        int component = 0;


        for (int bar = 0; bar < BARS; bar++) {
            for (int beat = 0; beat < BEATS; beat++) {
                int a = selectAction(s, 0);
                int[] s_dash = performAction(a, s);

                traj[component] = createKey(s_dash);

                component++;
                s = s_dash.clone();
            }
        }
        return traj;
    }

    public static void resetQ() {
        //resets the q values
        q = new HashMap[NUM_ACTIONS];

        for (int i = 0; i < q.length; i++) {
            q[i] = new HashMap();
        }
    }

    public static void playGame() {
        //int [] s = {-1, -1, -1, -1, -1, -1, -1, -1};
        int[] s = start_state;
        for (int bar = 0; bar < BARS; bar++) {
            for (int beat = 0; beat < BEATS; beat++) {
                int a = selectAction(s, 0);
                int[] s_dash = performAction(a, s);
                System.out.print(createKey(s_dash) + ", ");

                s = s_dash.clone();
            }
        }
    }

    public static String playSoftGame() {
        StringBuilder greedyP = new StringBuilder();
        String gP;
        int t = 1;

        int[] s = start_state;
        greedyP.append("([" + createKey(s) + "], ");
        for (int bar = 0; bar < BARS; bar++) {
            for (int beat = 0; beat < BEATS; beat++) {
                int a = selectSoftAction(s);

                /*System.out.print((bar + 1) + " " + (beat + 1) + "}-> ");
                for (int i = 0; i < NUM_ACTIONS; i++) {
                    System.out.print("(" + createKey(s) + ") " + i + " <-> " + q[i].get(createKey(s)) + " ");
                }
                System.out.println("|");*/

                greedyP.append(a + ")#");
                int[] s_dash = performAction(a, s);
                s = s_dash.clone();

                if (t != 64) {
                    greedyP.append("([" + createKey(s) + "], ");
                } else {
                    greedyP.append("([" + createKey(s) + "], -1)");
                }

                t++;
            }
            //System.out.println("");
        }

        gP = greedyP.toString();
        return gP;
    }

    public static String playGame2() {
        //int [] s = {-1, -1, -1, -1, -1, -1, -1, -1};
        StringBuilder greedyP = new StringBuilder();
        String gP;
        int t = 1;

        int[] s = start_state;
        greedyP.append("([" + createKey(s) + "], ");
        for (int bar = 0; bar < BARS; bar++) {
            for (int beat = 0; beat < BEATS; beat++) {
                int a = selectAction(s, 0);

                greedyP.append(a + ")#");
                int[] s_dash = performAction(a, s);
                s = s_dash.clone();

                if (t != 64) {
                    greedyP.append("([" + createKey(s) + "], ");
                } else {
                    greedyP.append("([" + createKey(s) + "], -1)");
                }

                t++;
            }
        }

        gP = greedyP.toString();
        return gP;
    }

    public static double updateQ(int[] s, int a, double r, int[] s_dash, int a_dash) {
        double val, val_dash, change, oldVal; //val = Q(s,a), val_dash = Q(s', a')

        String key = createKey(s);
        String key_dash = createKey(s_dash);

        val = (q[a].containsKey(key)) ? q[a].get(key) : 0; //q(s,a)
        val_dash = (q[a_dash].containsKey(key_dash)) ? q[a_dash].get(key_dash) : 0;//q(s',a')

        oldVal = val;

        double newVal = val + ALPHA * (r + GAMMA * val_dash - val);

        change = (oldVal - newVal != 0) ? Math.abs(oldVal - newVal) : Double.POSITIVE_INFINITY;
        //The above line: only if the difference is non-zero do we care about it. 
        //otherwise sarsa will stop on the first iteration because the first update backs up 0 to 0, and thus there is a 0 change.        

        q[a].remove(key);
        q[a].put(key, newVal);

        return change;
    }

    public static int[] performAction(int a, int[] s) {
        int[] new_s = s.clone();
        for (int i = s.length - 1; i > 0; i--) {
            new_s[i] = new_s[i - 1];
        }
        new_s[0] = a;

        return new_s;
    }

    public static int selectAction(int[] s, double epsilon) {
        int action = -1;
        Random r = new Random();
        double rand = r.nextDouble();

        action = (rand >= epsilon) ? greedyAction(s) : r.nextInt(NUM_ACTIONS);

        return action;
    }

    public static int selectSoftAction(int[] s) {
        int action = -1;
        String s_key = createKey(s);
        double denominator = 0;
        double[] probabilities = new double[NUM_ACTIONS];
        double[] cumul_probs = new double[NUM_ACTIONS];


        for (int i = 0; i < NUM_ACTIONS; i++) {
            double q_val = q[i].containsKey(s_key) ? q[i].get(s_key) : 0;
            //if (q_val < 0)
            //q_val = 0;
            double exponent = q_val / TEMP;
            denominator += Math.pow(Math.E, exponent);
        }

        //System.out.println("");
        //System.out.println("THE CURRENT STATE IS: " + s_key);
        
        for (int i = 0; i < NUM_ACTIONS; i++) {
            double q_val = q[i].containsKey(s_key) ? q[i].get(s_key) : 0;
            //if (q_val < 0)
            //q_val = 0;
            double exponent = q_val / TEMP;
            double numerator = Math.pow(Math.E, exponent);
            probabilities[i] = numerator / denominator;
            //System.out.print(" |action" + i + " prob = " + probabilities[i] + "| ");
        }
        //System.out.println("");

        for (int i = 0; i < probabilities.length; i++) {
            for (int j = 0; j <= i; j++) {
                cumul_probs[i] += probabilities[j];
            }

            //System.out.print(" |action" + i + " cumul prob = " + cumul_probs[i] + "| ");
        }
        //System.out.println();
        Random r = new Random();
        double num = r.nextDouble();
        //System.out.println("random num: " + num);
        for (int i = 0; i < probabilities.length; i++) {
            if (num >= (cumul_probs[i] - probabilities[i]) && num <= cumul_probs[i]) {
                return i;
            }
        }

        System.out.println("BAD - selectSoftAction() is broken...");

        return action;
    }

    public static int greedyAction(int[] s) {
        String s_key = createKey(s);
        int action = -2;
        double MAX = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < NUM_ACTIONS; i++) {
            boolean b = q[i].containsKey(s_key);
            double q_value = b ? q[i].get(s_key) : 0;

            if (q_value > MAX) {
                action = i;
                MAX = q_value;
            }
        }

        return action;
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

    //Q learning stuff
    public static void qLearningLambda(Map<String, Double> w) {
        int i = 0;

        try {
            PrintWriter out = new PrintWriter(new File("test.txt"));
            boolean nextEp = true;
            int policyCount = 0;
            String previousPol = "nothing";

            while (nextEp) {
                int[] s = start_state;
                int a = selectAction(s, EPSILON);

                //reset eligibility traces
                for (int trace = 0; trace < e.length; trace++) {
                    e[trace] = new HashMap<String, Double>();
                }

                //System.out.println("let's see");

                for (int bar = 0; bar < BARS; bar++) {
                    for (int beat = 0; beat < BEATS; beat++) {
                        //System.out.println("what happens");
                        //take action, get new state, s'
                        int[] s_dash;
                        s_dash = performAction(a, s);

                        out.println(createKey(s) + " - " + a + " : " + q[a].get(createKey(s)));

                        //observe r
                        String s_dash_key = createKey(s_dash);
                        String s_key = createKey(s);

                        double r = (w.containsKey(s_dash_key)) ? w.get(s_dash_key) : -5;
                        //get a' from s'
                        int a_dash = selectAction(s_dash, EPSILON);
                        //get a_star
                        int a_star = selectAction(s_dash, 0);
                        //get delta
                        double q_dash = q[a_star].containsKey(s_dash_key) ? q[a_star].get(s_dash_key) : 0;
                        double q_val = q[a].containsKey(s_key) ? q[a].get(s_key) : 0;
                        double delta = r + GAMMA * q_dash - q_val; // getQVal(s_dash, a_star) - getQVal(s, a);
                        //increment trace
                        incrementTrace(s, a);
                        //update Q val
                        updateQTrace(delta);
                        //degrade traces
                        degradeTraces();

                        //update states and actions
                        s = s_dash.clone();
                        a = a_dash;
                    }
                }
                i++;
                if (i % 1 == 0) {
                    out.println(i + "-----------------------");
                    String greedyP = playGame2();
                    if (previousPol.equals(greedyP)) {
                        policyCount++;
                    }

                    if (policyCount == 6000) {
                        nextEp = false;
                    }

                    previousPol = greedyP;
                }
                if (i % 1000 == 0) {
                    System.out.println("Episode Number: " + i);
                    //System.out.println(playGame2());
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (true) {
            System.out.println("Final Greedy: ");
            String greedyP = playGame2();
            System.out.println(greedyP);
            convert(greedyP);
        }
        if(true) {
            System.out.println("Final Softmax: ");
            String softP = playSoftGame();
            System.out.println(softP);
            convert(softP);
        }
    }

    public static void convert(String trajectory) {
        int[] traj = new int[64];
        int STATE_SIZE = 4;

        String[] statesAndAction = trajectory.split("#");

        for (int step = 0; step < statesAndAction.length; step++) {
            String string = statesAndAction[step];
            string = string.substring(1, string.length() - 1);
            String[] sa = string.split(",");
            sa[1] = sa[1].trim();
            sa[0] = sa[0].substring(1, sa[0].length() - 1);

            String[] stateString = sa[0].split(" ");
            int[] state = new int[STATE_SIZE];

            for (int k = 0; k < STATE_SIZE; k++) {
                state[k] = Integer.parseInt(stateString[k]);
            }

            int action = Integer.parseInt(sa[1]);

            if (action != -1) {
                traj[step] = action;
            }

            StateAction sa_obj = new StateAction(state, action);
        }

        int k = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                System.out.print(traj[k] + " ");
                k++;
            }
            System.out.println("");
        }
    }

    public static void updateQTrace(double delta) {
        for (int action = 0; action < NUM_ACTIONS; action++) {
            for (Map.Entry<String, Double> entry : e[action].entrySet()) {
                double trace = e[action].get(entry.getKey());
                boolean containsKey = q[action].containsKey(entry.getKey());
                double newQVal = containsKey ? q[action].get(entry.getKey()) + ALPHA * delta * trace : ALPHA * delta * e[action].get(entry.getKey());
                q[action].put(entry.getKey(), newQVal);
            }
        }
    }

    public static void degradeTraces() {
        for (int action = 0; action < NUM_ACTIONS; action++) {
            for (Map.Entry<String, Double> entry : e[action].entrySet()) {
                double newE = GAMMA * LAMBDA * e[action].get(entry.getKey());
                e[action].put(entry.getKey(), newE);
            }
        }
    }

    public static void incrementTrace(int[] s, int a) {
        String key = createKey(s);

        double val = e[a].containsKey(key) ? e[a].get(key) : 0;
        val += 1;
        e[a].put(key, val);
    }
}
