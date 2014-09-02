/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package tablebasedagent;

/**
 *
 * @author orrymr
 */
class StateAction {
    int [] state;
    int action;
    
    public StateAction(int [] state, int action){
        this.state = state;
        this.action = action;
    }
    
    public int [] getState(){
        return this.state;
    }
    
    public int getAction(){
        return this.action;
    }
}
