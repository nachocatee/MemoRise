import React from "react";
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  Switch,
  Pressable,
} from "react-native";
import Colors from "../constants/colors";

import { calculateDynamicWidth } from "../constants/dynamicSize";

type GroupSettingProps = {
  nameModalHandler: () => void;
  name: string;
  toggleSwitch: () => void;
  isEnabled: boolean;
  password: string;
  passwordModalHandler: () => void;
};

const screenWidth = Dimensions.get("window").width;

const GroupSetting: React.FC<GroupSettingProps> = ({
  nameModalHandler,
  name,
  toggleSwitch,
  isEnabled,
  password,
  passwordModalHandler,
}) => {
  return (
    <View style={styles.contentsContainer}>
      <View>
        <View style={styles.itemContainer}>
          <Text style={styles.text}>그룹 이름</Text>
          <Pressable onPress={nameModalHandler}>
            <Text style={[styles.text, { opacity: 0.5 }]}>{name}</Text>
          </Pressable>
        </View>

        <View style={styles.line}></View>

        <View style={styles.itemContainer}>
          <Text style={styles.text}>그룹 비공개</Text>
          <Switch
            trackColor={{ false: Colors.hover, true: Colors.primary300 }}
            thumbColor="white"
            onValueChange={toggleSwitch}
            value={isEnabled}
            style={{
              transform: [
                { scaleX: calculateDynamicWidth(1) },
                { scaleY: calculateDynamicWidth(1) },
              ],
            }}
          />
        </View>
        {password && (
          <>
            <View style={styles.line}></View>
            <View style={styles.itemContainer}>
              <Text style={styles.text}>비밀번호</Text>
              <Pressable onPress={passwordModalHandler}>
                <Text style={[styles.text, { opacity: 0.5 }]}>{password}</Text>
              </Pressable>
            </View>
          </>
        )}
      </View>
    </View>
  );
};

export default GroupSetting;

const styles = StyleSheet.create({
  contentsContainer: {
    flex: 1,
    alignItems: "center",
    marginTop: calculateDynamicWidth(25),
    justifyContent: "space-between",
  },
  itemContainer: {
    marginTop: calculateDynamicWidth(15),
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    width: screenWidth - 60,
  },
  text: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(18),
    color: Colors.text,
  },
  line: {
    borderBottomWidth: 1,
    borderBottomColor: Colors.hover,
    height: calculateDynamicWidth(30),
    width: screenWidth - 60,
    marginBottom: calculateDynamicWidth(10),
  },
});
