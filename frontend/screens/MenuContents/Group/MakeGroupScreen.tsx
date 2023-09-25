import React, { useState } from "react";
import { View, Text, Modal } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { BlurView } from "@react-native-community/blur";
import axios from "axios";
import { useSelector } from "react-redux";
import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";

import ConfirmBtn from "../../../components/Button/ConfirmBtn";
import GoBackHeader from "../../../components/Header/GoBackHeader";
import { styles } from "./GroupStyle";
import PasswordInputModal from "../../../components/Modal/Group/PasswordInputModal";
import GroupNameInputModal from "../../../components/Modal/Group/GroupNameInputModal";
import { RootState } from "../../../store/store";
import { BACKEND_URL } from "../../../util/http";
import GroupSetting from "../../../components/GroupSetting";

type RootStackParamList = {
  MakeGroup: undefined;
  GroupDetail: GroupDetailParams;
};

type GroupDetailParams = {
  teamSeq: number;
  userSeq: number;
};

const MakeGroupScreen = () => {
  const navigation =
    useNavigation<StackNavigationProp<RootStackParamList, "MakeGroup">>();
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // 그룹 생성 axios
  const GroupCreate = () => {
    axios({
      method: "POST",
      url: BACKEND_URL + "/teams",
      data: {
        name: name,
        owner: userId,
        // owner: 26, // 더미 데이터
        password: password,
      },
    })
      .then((res) => {
        console.log("그룹 생성 완료");
        navigation.navigate("GroupDetail", {
          teamSeq: res.data.teamSeq,
          userSeq: userId, // 더미 데이터
        });
      })
      .catch((err) => {
        console.log(err);
      });
  };

  // 그룹 이름 설정
  const [name, setName] = useState("내 그룹");
  const [enteredName, setEnteredName] = useState("");

  const nameModalHandler = () => {
    setNameModalVisible(true);
    setEnteredName("");
  };

  const nameInputHandler = (enterdText: string) => {
    setEnteredName(enterdText);
  };

  const nameConfirmHandler = () => {
    if (enteredName) {
      setName(enteredName);
      setNameModalVisible(false);
      setInvalid(false);
    } else {
      setInvalid(true);
    }
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
    setInvalid(false);
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

  return (
    <LinearGradient
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

        {/* 그룹 이름, 비밀번호 설정 */}
        <GroupSetting
          nameModalHandler={nameModalHandler}
          name={name}
          toggleSwitch={toggleSwitch}
          isEnabled={isEnabled}
          password={password}
          passwordModalHandler={passwordModalHandler}
        />

        <View style={styles.btnContainer}>
          <ConfirmBtn onPress={GroupCreate}>확인</ConfirmBtn>
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
                invalid={invalid}
              />
            </View>
          </Modal>
        </BlurView>
      )}
    </LinearGradient>
  );
};

export default MakeGroupScreen;
