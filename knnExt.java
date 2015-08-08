/**
 * Created by Merlin on 2015/5/20.
 */

package weka.classifiers.yym;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import java.util.Enumeration;

public class knnExt extends Classifier {

    private Instances m_Train;
    private double [] m_MinArray;
    private double [] m_MaxArray;

    public void buildClassifier(Instances instances) throws Exception {

        getCapabilities().testWithFail(instances);
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        m_Train = new Instances(instances, 0, instances.numInstances());
        //增加属性
        m_Train.insertAttributeAt(new Attribute("distance"), m_Train.numAttributes());
        m_Train.insertAttributeAt(new Attribute("angle"), m_Train.numAttributes());

        m_MinArray = new double [m_Train.numAttributes()];
        m_MaxArray = new double [m_Train.numAttributes()];
        for (int i = 0; i < m_Train.numAttributes(); i++) {
            m_MinArray[i] = m_MaxArray[i] = Double.NaN;
        }
        Enumeration enu = m_Train.enumerateInstances();
        while (enu.hasMoreElements()) {
            updateMinMax((Instance) enu.nextElement());
        }
    }

    public double classifyInstance(Instance instance) throws Exception {

        if (m_Train.numInstances() == 0) {
            throw new Exception("No training instances!");
        }

        double distance, angle, classWeightSum=0, classValue = 0;
        Instances m_MinAngel, m_MinDistance;

        updateMinMax(instance);

        Enumeration enu = m_Train.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance trainInstance = (Instance) enu.nextElement();
            if (!trainInstance.classIsMissing()) {
                distance = distance(instance, trainInstance);
                angle = angle(instance, trainInstance);
                //保存计算结果
                trainInstance.setValue(m_Train.attribute("distance"), distance);
                trainInstance.setValue(m_Train.attribute("angle"), angle);
            }
        }
        //按夹角排序
        m_Train.sort(m_Train.attribute("angle").index());
        //将前2个复制到m_MinAngle
        m_MinAngel = new Instances(m_Train, 0, 2);
        m_MinAngel.insertAttributeAt(new Attribute("classWeight"), m_MinAngel.numAttributes());
        m_Train.sort(m_Train.attribute("distance").index());
        m_MinDistance = new Instances(m_Train, 0, 2);
        m_MinDistance.insertAttributeAt(new Attribute("classWeight"), m_MinDistance.numAttributes());

        //将夹角和距离对象合并为一个新对象
        Instances m_Weight = new Instances(m_MinAngel);
        for (int i=0; i<m_MinDistance.numInstances()-1; i++){
            m_Weight.add(m_MinDistance.instance(i));
        }
        //将夹角归一化
        m_Weight.sort(m_Weight.attribute("angle").index());
        for (int i=0; i<m_Weight.numInstances()-1; i++){
            angle = m_Weight.instance(i).value(m_Weight.attribute("angle").index())/m_Weight.instance(m_Weight.numInstances()-1).value(m_Weight.attribute("angle").index());
            m_Weight.instance(i).setValue(m_Weight.attribute("angle").index(), angle);
        }
        //将距离归一化
        m_Weight.sort(m_Weight.attribute("distance").index());
        for (int i=0; i<m_Weight.numInstances()-1; i++){
            distance = m_Weight.instance(i).value(m_Weight.attribute("distance").index())/m_Weight.instance(m_Weight.numInstances()-1).value(m_Weight.attribute("distance").index());
            m_Weight.instance(i).setValue(m_Weight.attribute("distance").index(), distance);
        }
        //计算权值
        for (int i=0; i<m_Weight.numInstances()-1; i++){
            angle = m_Weight.instance(i).value(m_Weight.attribute("angle").index());
            distance = m_Weight.instance(i).value(m_Weight.attribute("distance").index());
            m_Weight.instance(i).setValue(m_Weight.attribute("classWeight").index(),100/((Math.abs(angle-distance)+1)*(angle+1)*(distance+100)));
        }
        //将权值归一化
        m_Weight.sort(m_Weight.attribute("classWeight").index());
        for (int i=0; i<m_Weight.numInstances()-1; i++){
            classWeightSum += m_Weight.instance(i).value(m_Weight.attribute("classWeight").index());
        }
        for (int i=0; i<m_Weight.numInstances()-1; i++){
            double classWeight = m_Weight.instance(i).value(m_Weight.attribute("classWeight").index());
            m_Weight.instance(i).setValue(m_Weight.attribute("classWeight").index(), classWeight/classWeightSum);
        }
        //计算classValue
        for (int i=0; i<m_Weight.numInstances()-1; i++){
            classValue += m_Weight.instance(i).value(m_Weight.attribute("classWeight").index()) * m_Weight.instance(i).classValue();
        }

        return classValue;
    }


    private double angle(Instance first, Instance second) {

        double diff = 0, diff1 = 0, diff2 = 0, angle, innerProduct = 0, moduleProduct, module1 = 0, module2 = 0;

        for(int i = 0; i < m_Train.attribute("distance").index(); i++) {
            if (i == m_Train.classIndex()) {
                continue;
            }
            if (!first.isMissing(i) && !second.isMissing(i)){
                diff = norm(first.value(i), i) * norm(second.value(i), i);
                diff1 = norm(first.value(i), i) * norm(first.value(i), i);
                diff2 = norm(second.value(i), i) * norm(second.value(i), i);
            }
            innerProduct += diff;
            module1 += diff1;
            module2 += diff2;
        }
        moduleProduct = Math.sqrt(module1) * Math.sqrt(module2);
        angle=Math.acos(innerProduct/moduleProduct);
        return angle;
    }

    private double distance(Instance first, Instance second) {

        double diff, distance = 0;

        for(int i = 0; i < m_Train.attribute("distance").index(); i++) {
            if (i == m_Train.classIndex()) {
                continue;
            }
            if (m_Train.attribute(i).isNominal()) {
                if (first.isMissing(i) || second.isMissing(i) ||
                        ((int)first.value(i) != (int)second.value(i))) {
                    distance += 1;
                }
            } else {
                if (first.isMissing(i) || second.isMissing(i)){
                    if (first.isMissing(i) && second.isMissing(i)) {
                        diff = 1;
                    } else {
                        if (second.isMissing(i)) {
                            diff = norm(first.value(i), i);
                        } else {
                            diff = norm(second.value(i), i);
                        }
                        if (diff < 0.5) {
                            diff = 1.0 - diff;
                        }
                    }
                } else {
                    diff = norm(first.value(i), i) - norm(second.value(i), i);
                }
                distance += diff * diff;
            }
        }
        return Math.sqrt(distance);
    }

    private double norm(double x,int i) {

        if (Double.isNaN(m_MinArray[i])
                || Utils.eq(m_MaxArray[i], m_MinArray[i])) {
            return 0;
        } else {
            return (x - m_MinArray[i]) / (m_MaxArray[i] - m_MinArray[i]);
        }
    }

    private void updateMinMax(Instance instance) {

        for (int j = 0;j < m_Train.attribute("distance").index(); j++) {
            if ((m_Train.attribute(j).isNumeric()) && (!instance.isMissing(j))) {
                if (Double.isNaN(m_MinArray[j])) {
                    m_MinArray[j] = instance.value(j);
                    m_MaxArray[j] = instance.value(j);
                } else {
                    if (instance.value(j) < m_MinArray[j]) {
                        m_MinArray[j] = instance.value(j);
                    } else {
                        if (instance.value(j) > m_MaxArray[j]) {
                            m_MaxArray[j] = instance.value(j);
                        }
                    }
                }
            }
        }
    }

    public static void main(String [] argv) {
        runClassifier(new knnExt(), argv);
    }
}
