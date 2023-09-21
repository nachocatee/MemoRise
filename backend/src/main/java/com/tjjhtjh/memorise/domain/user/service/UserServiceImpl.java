package com.tjjhtjh.memorise.domain.user.service;

import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.exception.UserEmailDuplicateException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.Role;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.dto.request.JoinRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.request.UpdateUserInfoRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional(readOnly = true)
@RequiredArgsConstructor
@Slf4j
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;

    @Override
    @Transactional
    public User join(JoinRequest joinRequest) {
        if(userRepository.findByEmailAndIsDeletedFalse(joinRequest.getEmail()) != null) {
            throw new UserEmailDuplicateException("이미 존재하는 이메일입니다.");
        }
        User user = new User(joinRequest.getEmail(), joinRequest.getNickname(), joinRequest.getProfile(), Role.MEMBER);
        return userRepository.save(user);
    }

    @Override
    @Transactional
    public void updateUserInfo(Long userSeq, UpdateUserInfoRequest updateUserInfoRequest) {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq);
        if (user == null) {
            throw new NoUserException("존재하지 않는 유저입니다.");
        }
        user.update(userSeq, updateUserInfoRequest.getNickname(), updateUserInfoRequest.getProfile());
        userRepository.save(user);
    }

    @Override
    public User getUserInfo(Long userSeq) {
        User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq);
        if (user == null) {
            throw new NoUserException("존재하지 않는 유저입니다.");
        }
        return user;
    }

}
