import React, { useState } from "react";
import { View, Text, Pressable, Switch, Modal } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { BlurView } from "@react-native-community/blur";

import ConfirmBtn from "../../../components/Button/ConfirmBtn";
import GoBackHeader from "../../../components/Header/GoBackHeader";
import Colors from "../../../constants/colors";
import { styles } from "./GroupStyle";
import PasswordInputModal from "../../../components/Modal/PasswordInputModal";
import GroupNameInputModal from "../../../components/Modal/GroupNameInputModal";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";

const MakeGroupScreen = () => {
  // 그룹 이름 설정
  const [name, setName] = useState("Group123"); // 더미 데이터
  const [enteredName, setEnteredName] = useState("");

  const nameModalHandler = () => {
    setNameModalVisible(true);
    setEnteredName("");
  };

  const nameInputHandler = (enterdText: string) => {
    setEnteredName(enterdText);
  };

  const nameConfirmHandler = () => {
    setName(enteredName);
    setNameModalVisible(false);
  };

  // 비밀번호 설정
  const [password, setPassword] = useState("");
  const [enteredPassword, setEnteredPassword] = useState("");
  const [invalid, setInvalid] = useState(false);

  const passwordModalHandler = () => {
    setPasswordModalVisible(true);
    setEnteredPassword("");
  };

  const passwordInputHandler = (enteredText: string) => {
    setEnteredPassword(enteredText);
  };

  const passwordConfirmHandler = () => {
    if (enteredPassword.length < 4) {
      setInvalid(true);
      return;
    }
    setPassword(enteredPassword);
    setPasswordModalVisible(false);
    setInvalid(false);
  };

  // 모달 상태
  const [isPasswordModalVisible, setPasswordModalVisible] = useState(false);
  const [isNameModalVisible, setNameModalVisible] = useState(false);

  const closePasswordModal = () => {
    if (password) {
      setPasswordModalVisible(false);
      setInvalid(false);
    } else {
      setPasswordModalVisible(false);
      setIsEnabled(false);
      setInvalid(false);
    }
  };

  const closeNameModal = () => {
    setNameModalVisible(false);
  };

  // 토글 스위치
  const [isEnabled, setIsEnabled] = useState(false);

  const toggleSwitch = () => {
    setIsEnabled((previousState) => {
      if (!previousState) {
        // isEnabled가 현재 false인 경우 (즉, true로 바뀔 경우)
        passwordModalHandler();
      } else {
        setPassword("");
      }
      return !previousState;
    });
  };

  const confirmHandler = () => {};

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
                    <Text style={[styles.text, { opacity: 0.5 }]}>
                      {password}
                    </Text>
                  </Pressable>
                </View>
              </>
            )}
          </View>
        </View>
        {/* 여기까지 */}

        <View style={styles.btnContainer}>
          <ConfirmBtn onPress={confirmHandler}>확인</ConfirmBtn>
        </View>
      </View>

      {/* 비밀번호 모달 */}
      {isPasswordModalVisible && (
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
            visible={isPasswordModalVisible}
            onRequestClose={closePasswordModal}
          >
            <View style={styles.modalContainer}>
              <PasswordInputModal
                closeModal={closePasswordModal}
                onChangeText={passwordInputHandler}
                value={enteredPassword}
                onConfirm={passwordConfirmHandler}
                invalid={invalid}
              />
            </View>
          </Modal>
        </BlurView>
      )}

      {/* 그룹이름 모달 */}
      {isNameModalVisible && (
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
            visible={isNameModalVisible}
            onRequestClose={closeNameModal}
          >
            <View style={styles.modalContainer}>
              <GroupNameInputModal
                closeModal={closeNameModal}
                onChangeText={nameInputHandler}
                value={enteredName}
                onConfirm={nameConfirmHandler}
              />
            </View>
          </Modal>
        </BlurView>
      )}
    </LinearGradient>
  );
};

export default MakeGroupScreen;
