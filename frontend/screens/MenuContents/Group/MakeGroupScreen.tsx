import React, { useState } from "react";
import { View, Text, Image, Pressable, Switch } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import ConfirmBtn from "../../../components/Button/ConfirmBtn";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import Colors from "../../../constants/colors";
import { styles } from "./GroupStyle";

const MakeGroupScreen = () => {
  // 모달 상태
  const [isModalVisible, setModalVisible] = useState(false);

  // 토글 스위치
  const [isEnabled, setIsEnabled] = useState(false);
  const toggleSwitch = () => {
    setIsEnabled((previousState) => !previousState);
    setModalVisible(true);
  };

  const closeModal = () => {
    setModalVisible(false);
  };

  const ConfirmHandler = () => {};

  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        <View style={styles.titleContainer}>
          <Text style={styles.title}>그룹 만들기</Text>
        </View>

        {/* 여기서부터 나중에 컴포넌트로 뺄거임 */}
        <View style={styles.contentsContainer}>
          <View>
            <View style={styles.itemContainer}>
              <Text style={styles.text}>그룹 이름</Text>
              <Pressable>
                <Text style={[styles.text, { opacity: 0.5 }]}>Group123</Text>
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
              />
            </View>
          </View>
        </View>
        {/* 여기까지 */}

        <View style={styles.btnContainer}>
          <ConfirmBtn onPress={ConfirmHandler}>확인</ConfirmBtn>
        </View>
      </View>
    </LinearGradient>
  );
};

export default MakeGroupScreen;
