import React, { useState } from "react";
import { View, Text, Pressable, Switch, Modal } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import ConfirmBtn from "../../../components/Button/ConfirmBtn";
import { BlurView } from "@react-native-community/blur";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import Colors from "../../../constants/colors";
import { styles } from "./GroupStyle";
import PasswordInputModal from "../../../components/Modal/PasswordInputModal";

const MakeGroupScreen = () => {
  // 비밀번호 설정
  const [password, setPassword] = useState("");

  // 모달 상태
  const [isModalVisible, setModalVisible] = useState(false);

  // 토글 스위치
  const [isEnabled, setIsEnabled] = useState(false);

  const toggleSwitch = () => {
    setIsEnabled((previousState) => {
      if (!previousState) {
        // isEnabled가 현재 false인 경우 (즉, true로 바뀔 경우)
        setModalVisible(true);
      }
      return !previousState;
    });
  };

  const closeModal = () => {
    setModalVisible(false);
    setIsEnabled(false);
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

      {/* 모달 */}
      {isModalVisible && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
          blurType="light"
          blurAmount={4}
        >
          <Modal
            transparent={true}
            animationType="fade"
            visible={isModalVisible}
            onRequestClose={closeModal}
          >
            <View style={styles.modalContainer}>
              <PasswordInputModal closeModal={closeModal} />
            </View>
          </Modal>
        </BlurView>
      )}
    </LinearGradient>
  );
};

export default MakeGroupScreen;
