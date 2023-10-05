package com.tjjhtjh.memorise.domain.item.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.item.repository.entity.Item;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;

import java.util.List;

import static com.tjjhtjh.memorise.domain.item.repository.entity.QItem.item;

public class ItemRepositoryImpl  extends QuerydslRepositorySupport implements ItemRepositoryCustom {
    private final JPAQueryFactory queryFactory;

    public ItemRepositoryImpl(JPAQueryFactory jpaQueryFactory){
        super(Item.class);
        this.queryFactory = jpaQueryFactory;
    }


    @Override
    public List<Long> itemSeqList() {
        return queryFactory.select(item.itemSeq).from(item).fetch();
    }
}
