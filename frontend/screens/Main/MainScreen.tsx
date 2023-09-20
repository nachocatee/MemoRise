import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import React, { useState } from "react";
import { View, Text, Modal, Pressable, Image } from "react-native";
import { BlurView } from "@react-native-community/blur";

import ConfirmBtn from "../../components/Button/ConfirmBtn";
import MainHeader from "../../components/Header/MainHeader";
import { styles } from "./MainStyle";

type RootStackParamList = {
  SignUp: undefined;
};

const MainScreen = () => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();

  // 임시 네비게이터
  const ImsiHandler = () => {
    navigation.navigate("SignUp");
  };

  // 알림 모달
  const [isModalVisible, setModalVisible] = useState(false);

  const closeModal = () => {
    setModalVisible(false);
  };

  return (
    <View>
      {!isModalVisible && (
        <MainHeader openModal={() => setModalVisible(true)} />
      )}
      <View>
        <ConfirmBtn onPress={ImsiHandler}>뒤로가기</ConfirmBtn>
      </View>

      {/* 알림 모달 */}
      {isModalVisible && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
          blurType="dark"
          blurAmount={4}
        >
          <Modal
            transparent={true}
            animationType="slide"
            visible={isModalVisible}
            onRequestClose={closeModal}
          >
            {/* 헤더 */}
            <View style={styles.header}>
              <Pressable>
                <Image
                  source={require("../../assets/image/logo/logowhite.png")}
                  style={styles.logo}
                />
              </Pressable>
              <Pressable style={styles.cancelContainer} onPress={closeModal}>
                <Image
                  source={require("../../assets/icons/cancelwhite.png")}
                  style={styles.cancel}
                />
              </Pressable>
            </View>

            <View style={styles.modalEmptyContainer}>
              <Text style={styles.modalEmpty}>알림 없음</Text>
            </View>
          </Modal>
        </BlurView>
      )}
    </View>
  );
};

export default MainScreen;
